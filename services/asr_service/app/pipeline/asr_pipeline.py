"""ASR pipeline — audio chunks → VAD → transcription → partials → final.

One pipeline instance per session.  Caller pushes AudioChunk objects;
the pipeline emits PartialTranscript and FinalTranscript via callbacks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

from glossa_common.logging import get_logger

from ..domain.audio_buffer import ChunkedAudioBuffer, RingAudioBuffer
from ..domain.entities import (
    ASRSession,
    AudioChunk,
    FinalTranscript,
    PartialTranscript,
    TranscriptSegment,
    TranscriptType,
    WordTimestamp,
)
from ..domain.interfaces import MedicalAdapterPort, TranscriberPort, VADPort
from .partial_emitter import PartialEmitter

logger = get_logger(__name__)

OnPartial = Callable[[PartialTranscript], Coroutine[Any, Any, None]]
OnFinal   = Callable[[FinalTranscript],   Coroutine[Any, Any, None]]


class ASRPipeline:
    """Full streaming ASR pipeline for one session.

    Architecture::

        push_chunk()
            ↓
        [RingAudioBuffer]  ← raw PCM accumulation
            ↓  (ready_to_emit)
        [VAD]              ← speech segment detection
            ↓  (speech segments)
        [ChunkedAudioBuffer] ← Whisper-sized window assembly
            ↓  (30 s chunks or flush)
        [FasterWhisperRunner.transcribe_streaming()]
            ↓  (TranscriptSegment per decoded phrase)
        [PartialEmitter]   ← deduplicated partial emission
            ↓  on_partial callback
        [MedicalAdapter]   ← terminology correction on final
            ↓  on_final callback
    """

    def __init__(
        self,
        session: ASRSession,
        transcriber: TranscriberPort,
        vad: VADPort,
        medical_adapter: MedicalAdapterPort,
        on_partial: OnPartial,
        on_final: OnFinal,
        emit_partials: bool = True,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = False,
    ) -> None:
        self._session = session
        self._transcriber = transcriber
        self._vad = vad
        self._medical = medical_adapter
        self._on_partial = on_partial
        self._on_final = on_final
        self._emit_partials = emit_partials
        self._language = language or session.language
        self._beam_size = beam_size
        self._word_timestamps = word_timestamps

        sr = session.sample_rate
        self._ring = RingAudioBuffer(
            capacity_s=60.0,
            sample_rate=sr,
            overlap_s=0.5,
            emit_threshold_s=0.5,
        )
        self._chunked = ChunkedAudioBuffer(
            sample_rate=sr,
            max_chunk_s=30.0,
            min_chunk_s=0.3,
            overlap_s=2.0,
        )
        self._emitter = PartialEmitter(
            session_id=session.session_id,
            request_id="",
            language=self._language,
        )
        self._first_chunk_at: float | None = None
        self._active_request_id: str = ""
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def push_chunk(self, chunk: AudioChunk) -> None:
        """Accept a raw audio chunk and run the pipeline."""
        if self._first_chunk_at is None:
            self._first_chunk_at = time.perf_counter()

        audio_f32 = chunk.to_float32()
        self._ring.push(audio_f32)
        self._session.total_audio_s += chunk.duration_s
        self._session.chunk_count += 1
        self._session.touch()

        if not self._ring.ready_to_emit:
            return

        await self._process_ring()

    async def flush(self) -> FinalTranscript | None:
        """Flush remaining buffered audio and emit a final transcript."""
        partial = self._ring.flush()
        if partial is not None and len(partial) > 0:
            await self._process_audio(partial)

        remainder = self._chunked.flush()
        if remainder is not None:
            return await self._transcribe_final(remainder)
        return None

    async def reset(self) -> None:
        """Reset pipeline state (new utterance)."""
        async with self._lock:
            self._ring.reset()
            self._chunked.reset()
            self._emitter.reset()
            self._vad.reset()
            self._first_chunk_at = None

    # ── Private ───────────────────────────────────────────────────────────────

    async def _process_ring(self) -> None:
        audio = self._ring.flush()
        if audio is None or len(audio) == 0:
            return
        await self._process_audio(audio)

    async def _process_audio(self, audio: np.ndarray) -> None:
        """Run VAD → chunk assembly → streaming transcription."""
        sr = self._session.sample_rate

        # VAD
        vad_segments = self._vad.process(audio, sr)
        if not vad_segments:
            return

        for vad_seg in vad_segments:
            chunks = self._chunked.push(vad_seg.audio)
            for chunk_audio in chunks:
                await self._transcribe_chunk(chunk_audio)

    async def _transcribe_chunk(self, audio: np.ndarray) -> None:
        """Stream-transcribe one Whisper window and emit partials."""
        t0 = time.perf_counter()
        try:
            async for seg in self._transcriber.transcribe_streaming(
                audio,
                sample_rate=self._session.sample_rate,
                language=self._language,
                beam_size=self._beam_size,
            ):
                if seg.no_speech_prob > 0.9:
                    continue
                partial = self._emitter.push_segment(seg)
                if partial and self._emit_partials:
                    await self._on_partial(partial)

        except Exception:
            logger.exception("pipeline_transcribe_chunk_error", session_id=self._session.session_id)

    async def _transcribe_final(self, audio: np.ndarray) -> FinalTranscript:
        """Run full transcription on the remaining audio, emit final transcript."""
        t0 = time.perf_counter()
        try:
            segments = await self._transcriber.transcribe(
                audio,
                sample_rate=self._session.sample_rate,
                language=self._language,
                beam_size=self._beam_size,
            )
        except Exception:
            logger.exception("pipeline_transcribe_final_error", session_id=self._session.session_id)
            segments = []

        inference_ms = (time.perf_counter() - t0) * 1000
        text = " ".join(s.text for s in segments if s.text).strip()

        corrected, medical_terms = self._medical.adapt(text, self._language)

        all_words: list[WordTimestamp] = []
        for seg in segments:
            all_words.extend(seg.words)

        total_latency_ms = (
            (time.perf_counter() - self._first_chunk_at) * 1000
            if self._first_chunk_at else 0.0
        )

        final = FinalTranscript(
            session_id=self._session.session_id,
            request_id=self._active_request_id,
            text=text,
            language=self._language,
            segments=segments,
            words=all_words,
            audio_duration_s=len(audio) / self._session.sample_rate,
            first_chunk_at=self._first_chunk_at or 0.0,
            inference_ms=inference_ms,
            total_latency_ms=total_latency_ms,
            corrected_text=corrected,
            medical_terms_found=medical_terms,
        )

        self._session.transcript_count += 1
        self._emitter.reset()
        self._first_chunk_at = None

        logger.info(
            "pipeline_final_transcript",
            session_id=self._session.session_id,
            text_len=len(text),
            inference_ms=round(inference_ms, 1),
            total_latency_ms=round(total_latency_ms, 1),
            medical_terms=len(medical_terms),
        )

        await self._on_final(final)
        return final
