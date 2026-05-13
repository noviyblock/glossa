"""Silero TTS synthesis engine — production-grade with warmup and CPU optimization.

Design:
- Lazy model load: first call to ensure_loaded() downloads/loads the model
- CPU thread pinning: torch.set_num_threads(cpu_threads) to avoid over-subscription
- Per-speaker model reuse: one model instance shared across all requests
- Warmup: synthesizes N short phrases to fill Torch JIT cache before serving traffic
- XTTS-compatible interface via VoiceProfile abstraction
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

from ..domain.entities import AudioFormat, BenchmarkResult, VoiceProfile
from ..domain.interfaces import SynthesizerPort

logger = get_logger(__name__)

# ── Built-in Russian voice profiles ──────────────────────────────────────────
_RU_VOICES: dict[str, VoiceProfile] = {
    "aidar": VoiceProfile(
        id="aidar", display_name="Айдар", speaker_key="aidar",
        language="ru", gender="male", engine="silero", sample_rate=24000,
    ),
    "baya": VoiceProfile(
        id="baya", display_name="Бая", speaker_key="baya",
        language="ru", gender="female", engine="silero", sample_rate=24000,
    ),
    "kseniya": VoiceProfile(
        id="kseniya", display_name="Ксения", speaker_key="kseniya",
        language="ru", gender="female", engine="silero", sample_rate=24000,
    ),
    "xenia": VoiceProfile(
        id="xenia", display_name="Ксения (x)", speaker_key="xenia",
        language="ru", gender="female", engine="silero", sample_rate=24000,
    ),
    "random": VoiceProfile(
        id="random", display_name="Случайный", speaker_key="random",
        language="ru", gender="unknown", engine="silero", sample_rate=24000,
    ),
}

# Phrases used to warm up JIT compiled model graph
_WARMUP_PHRASES = [
    "Добрый день.",
    "Как я могу вам помочь?",
    "Пожалуйста, повторите.",
    "Спасибо за обращение.",
    "Ваш запрос обрабатывается.",
]


class SileroTTSRunner(SynthesizerPort):
    """Full-featured Silero TTS runner with warmup, CPU opts, and XTTS-ready interface."""

    def __init__(
        self,
        model_dir: str | Path = "/models/silero",
        language: str = "ru",
        device: str = "cpu",
        cpu_threads: int = 4,
        warmup_on_load: bool = True,
        warmup_voice_id: str = "aidar",
    ) -> None:
        self._model_dir = Path(model_dir)
        self._language = language
        self._device_name = device
        self._cpu_threads = cpu_threads
        self._warmup_on_load = warmup_on_load
        self._warmup_voice_id = warmup_voice_id
        self._model: Any = None
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ensure_loaded(self) -> None:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
                if self._warmup_on_load:
                    await self.warmup()

    async def warmup(self) -> None:
        """Synthesize short phrases to pre-compile the Torch JIT graph."""
        voice = _RU_VOICES.get(self._warmup_voice_id, _RU_VOICES["aidar"])
        t0 = time.perf_counter()
        for phrase in _WARMUP_PHRASES:
            try:
                await asyncio.to_thread(self._synthesize_sync, phrase, voice, True, True)
            except Exception:
                logger.exception("warmup_phrase_failed", phrase=phrase[:40])
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("tts_warmup_done", phrases=len(_WARMUP_PHRASES), elapsed_ms=round(elapsed, 1))

    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        put_accent: bool = True,
        put_yo: bool = True,
    ) -> np.ndarray:
        await self.ensure_loaded()
        with MODEL_INFERENCE_LATENCY.labels(service="tts-service", model="silero").time():
            pcm = await asyncio.to_thread(self._synthesize_sync, text, voice, put_accent, put_yo)
        return pcm

    async def synthesize_sentences(
        self,
        sentences: list[str],
        voice: VoiceProfile,
        put_accent: bool = True,
        put_yo: bool = True,
    ) -> AsyncIterator[tuple[str, np.ndarray]]:
        await self.ensure_loaded()
        for sentence in sentences:
            if not sentence.strip():
                continue
            pcm = await asyncio.to_thread(
                self._synthesize_sync, sentence, voice, put_accent, put_yo
            )
            yield sentence, pcm

    def list_voices(self) -> list[VoiceProfile]:
        return list(_RU_VOICES.values())

    def get_voice(self, voice_id: str) -> VoiceProfile:
        return _RU_VOICES.get(voice_id, _RU_VOICES["aidar"])

    async def benchmark(
        self, text: str, voice: VoiceProfile, n: int = 10
    ) -> BenchmarkResult:
        await self.ensure_loaded()
        latencies: list[float] = []
        audio_duration = 0.0

        for _ in range(n):
            t0 = time.perf_counter()
            pcm = await asyncio.to_thread(self._synthesize_sync, text, voice, True, True)
            latencies.append((time.perf_counter() - t0) * 1000)
            audio_duration = len(pcm) / voice.sample_rate

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(0.95 * n)]
        p99 = latencies[int(0.99 * n)] if n >= 100 else latencies[-1]
        rtf = (p50 / 1000) / audio_duration if audio_duration > 0 else 0

        return BenchmarkResult(
            n_runs=n,
            text_len=len(text),
            voice_id=voice.id,
            sample_rate=voice.sample_rate,
            p50_ms=round(p50, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
            min_ms=round(min(latencies), 2),
            max_ms=round(max(latencies), 2),
            rtf=round(rtf, 4),
            chars_per_sec=round(len(text) / (p50 / 1000), 1) if p50 > 0 else 0,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        import torch

        # CPU thread optimization — set before any model ops
        if self._device_name == "cpu" and self._cpu_threads > 0:
            torch.set_num_threads(self._cpu_threads)
            torch.set_num_interop_threads(max(1, self._cpu_threads // 2))

        device = torch.device(self._device_name)
        local_path = self._model_dir / f"silero_tts_{self._language}.pt"

        if local_path.exists():
            model = torch.package.PackageImporter(str(local_path)).load_pickle(
                "tts_models", "model"
            )
            logger.info("silero_loaded_from_disk", path=str(local_path))
        else:
            logger.info("silero_downloading", language=self._language)
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=self._language,
                speaker=f"v4_{self._language}",
                trust_repo=True,
            )
            self._model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model, str(local_path))
            logger.info("silero_saved", path=str(local_path))

        model.to(device)
        model.eval()
        logger.info(
            "silero_ready",
            language=self._language,
            device=self._device_name,
            cpu_threads=self._cpu_threads,
        )
        return model

    def _synthesize_sync(
        self,
        text: str,
        voice: VoiceProfile,
        put_accent: bool,
        put_yo: bool,
    ) -> np.ndarray:
        import torch

        with torch.no_grad():
            audio_tensor = self._model.apply_tts(
                text=text,
                speaker=voice.speaker_key,
                sample_rate=voice.sample_rate,
                put_accent=put_accent,
                put_yo=put_yo,
            )

        if hasattr(audio_tensor, "numpy"):
            return audio_tensor.numpy().astype(np.float32)
        return np.array(audio_tensor, dtype=np.float32)
