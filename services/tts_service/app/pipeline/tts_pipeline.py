"""TTS pipeline — high-level orchestrator tying synthesis, cache, and encoding.

Flow (synthesize):
  text → normalize → cache lookup → [queue synthesis] → encode → cache store → return

Flow (synthesize_streaming):
  text → normalize → split sentences → per-sentence:
      cache lookup → [direct synthesis] → encode chunk → yield

The pipeline does NOT call the queue for streaming to avoid head-of-line blocking.
Streaming always synthesizes sentences directly.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import numpy as np

from glossa_common.logging import get_logger

from ..audio.encoder import AudioEncoder
from ..cache.phrase_cache import TieredPhraseCache, cache_key
from ..domain.entities import (
    AudioChunk,
    AudioFormat,
    BenchmarkResult,
    PhraseCacheEntry,
    SynthesisRequest,
    SynthesisResult,
    VoiceProfile,
)
from ..domain.interfaces import SynthesizerPort
from ..domain.text_processor import TextPreprocessor
from ..queue.synthesis_queue import SynthesisQueue, QueueFullError

logger = get_logger(__name__)

# Phrases pre-loaded into the cache on startup
_PRELOAD_PHRASES_RU: list[str] = [
    "Добро пожаловать.",
    "Как я могу вам помочь?",
    "Пожалуйста, подождите.",
    "Ваш запрос обрабатывается.",
    "Спасибо за обращение.",
    "К сожалению, я не понял вопрос.",
    "Пожалуйста, повторите.",
    "Соединяю с оператором.",
    "Обратитесь к врачу.",
    "Данные сохранены.",
]


class TTSPipeline:
    """Wires synthesizer, cache, encoder and queue into the service API."""

    def __init__(
        self,
        synthesizer: SynthesizerPort,
        cache: TieredPhraseCache,
        encoder: AudioEncoder,
        queue: SynthesisQueue,
        preprocessor: TextPreprocessor,
    ) -> None:
        self._synth = synthesizer
        self._cache = cache
        self._enc = encoder
        self._queue = queue
        self._pre = preprocessor

    # ── Public API ─────────────────────────────────────────────────────────────

    async def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        """Full synthesis → encoded bytes. Uses queue for backpressure."""
        t_start = time.perf_counter()

        voice = self._synth.get_voice(req.voice_id)
        normalized = self._pre.normalize(req.text)
        key = cache_key(normalized, voice.id, req.sample_rate, req.format)

        # L1/L2 cache lookup
        cached = await self._cache.get(key)
        if cached is not None:
            logger.debug("tts_cache_hit", key=key[:8], chars=len(normalized))
            return SynthesisResult(
                request_id=req.request_id,
                session_id=req.session_id,
                audio_bytes=cached.audio_bytes,
                format=cached.format,
                sample_rate=cached.sample_rate,
                duration_s=cached.duration_s,
                inference_ms=0.0,
                total_ms=(time.perf_counter() - t_start) * 1000,
                char_count=len(normalized),
                from_cache=True,
                voice_id=voice.id,
            )

        # Synthesize via queue
        t_inf = time.perf_counter()
        try:
            pcm = await self._queue.submit(
                text=normalized,
                voice=voice,
                put_accent=req.put_accent,
                put_yo=req.put_yo,
                priority=req.priority,
            )
        except QueueFullError:
            # Fallback: synthesize directly (bypass queue)
            logger.warning("queue_full_synthesizing_directly", chars=len(normalized))
            pcm = await self._synth.synthesize(normalized, voice, req.put_accent, req.put_yo)

        inference_ms = (time.perf_counter() - t_inf) * 1000

        # Encode
        audio_bytes = self._enc.encode(pcm, req.sample_rate, req.format)
        duration_s = self._enc.audio_duration(pcm, req.sample_rate)

        # Store in cache
        entry = PhraseCacheEntry(
            key=key,
            audio_bytes=audio_bytes,
            format=req.format,
            sample_rate=req.sample_rate,
            duration_s=duration_s,
            char_count=len(normalized),
        )
        await self._cache.set(entry)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "tts_synthesized",
            chars=len(normalized),
            voice=voice.id,
            format=req.format,
            inference_ms=round(inference_ms, 1),
            total_ms=round(total_ms, 1),
        )
        return SynthesisResult(
            request_id=req.request_id,
            session_id=req.session_id,
            audio_bytes=audio_bytes,
            format=req.format,
            sample_rate=req.sample_rate,
            duration_s=duration_s,
            inference_ms=inference_ms,
            total_ms=total_ms,
            char_count=len(normalized),
            from_cache=False,
            voice_id=voice.id,
        )

    async def synthesize_streaming(
        self, req: SynthesisRequest
    ) -> AsyncIterator[AudioChunk]:
        """Split text → synthesize each sentence → yield AudioChunk per sentence."""
        voice = self._synth.get_voice(req.voice_id)
        normalized = self._pre.normalize(req.text)
        sentences = self._pre.split_sentences(normalized)

        if not sentences:
            return

        total = len(sentences)
        for idx, sentence in enumerate(sentences):
            is_final = idx == total - 1
            key = cache_key(sentence, voice.id, req.sample_rate, req.format)

            # Try cache first for each sentence
            cached = await self._cache.get(key)
            if cached is not None:
                pcm_bytes = cached.audio_bytes
                duration_s = cached.duration_s
            else:
                # Direct synthesis (no queue — streaming must be low-latency)
                pcm = await self._synth.synthesize(
                    sentence, voice, req.put_accent, req.put_yo
                )
                pcm_bytes = self._enc.encode(pcm, req.sample_rate, req.format)
                duration_s = self._enc.audio_duration(pcm, req.sample_rate)
                # Cache for future
                await self._cache.set(PhraseCacheEntry(
                    key=key,
                    audio_bytes=pcm_bytes,
                    format=req.format,
                    sample_rate=req.sample_rate,
                    duration_s=duration_s,
                    char_count=len(sentence),
                ))

            yield AudioChunk(
                request_id=req.request_id,
                chunk_index=idx,
                audio_bytes=pcm_bytes,
                duration_s=duration_s,
                is_final=is_final,
                sentence_text=sentence,
            )

    async def preload_phrases(
        self,
        phrases: list[str] | None = None,
        voice_id: str = "aidar",
        fmt: AudioFormat = AudioFormat.WAV,
        sample_rate: int = 24000,
    ) -> int:
        """Warm up cache with frequently-used phrases. Returns number loaded."""
        phrases = phrases or _PRELOAD_PHRASES_RU
        voice = self._synth.get_voice(voice_id)
        loaded = 0
        for phrase in phrases:
            normalized = self._pre.normalize(phrase)
            key = cache_key(normalized, voice.id, sample_rate, fmt)
            existing = await self._cache.get(key)
            if existing is not None:
                continue
            try:
                pcm = await self._synth.synthesize(normalized, voice)
                audio_bytes = self._enc.encode(pcm, sample_rate, fmt)
                duration_s = self._enc.audio_duration(pcm, sample_rate)
                await self._cache.set(PhraseCacheEntry(
                    key=key,
                    audio_bytes=audio_bytes,
                    format=fmt,
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                    char_count=len(normalized),
                ))
                loaded += 1
            except Exception:
                logger.exception("preload_phrase_failed", phrase=phrase[:40])

        logger.info("phrase_preload_done", loaded=loaded, total=len(phrases))
        return loaded

    async def benchmark(
        self,
        text: str,
        voice_id: str = "aidar",
        n: int = 10,
        fmt: AudioFormat = AudioFormat.WAV,
    ) -> BenchmarkResult:
        voice = self._synth.get_voice(voice_id)
        result = await self._synth.benchmark(text, voice, n)
        result.format = fmt
        return result
