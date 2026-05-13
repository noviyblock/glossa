"""ASR benchmark service — latency profiling and RTF measurement."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import ASRBenchmarkResult
from ..domain.interfaces import TranscriberPort

logger = get_logger(__name__)


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    sorted_s = sorted(samples)
    idx = int(len(sorted_s) * p / 100)
    return sorted_s[min(idx, len(sorted_s) - 1)]


class ASRBenchmarkService:
    """Run latency and RTF benchmarks on a WhisperRunner instance."""

    def __init__(
        self,
        transcriber: TranscriberPort,
        sample_rate: int = 16000,
    ) -> None:
        self._transcriber = transcriber
        self._sr = sample_rate

    async def run(
        self,
        audio_duration_s: float = 5.0,
        n_samples: int = 20,
        language: str = "ru",
        beam_size: int = 5,
        warmup_iters: int = 3,
    ) -> ASRBenchmarkResult:
        """Generate synthetic audio and benchmark transcription latency."""
        await self._transcriber.ensure_loaded()

        # Warmup
        logger.info("asr_benchmark_warmup", iters=warmup_iters)
        await self._transcriber.warmup(warmup_iters)

        # Generate deterministic "silence" audio — fast to transcribe, consistent
        n_audio_samples = int(audio_duration_s * self._sr)
        dummy = np.zeros(n_audio_samples, dtype=np.float32)
        # Add very small noise so Whisper doesn't short-circuit
        dummy += np.random.randn(n_audio_samples).astype(np.float32) * 0.001

        logger.info(
            "asr_benchmark_start",
            audio_s=audio_duration_s,
            n_samples=n_samples,
            language=language,
        )

        latencies: list[float] = []
        for i in range(n_samples):
            t0 = time.perf_counter()
            await self._transcriber.transcribe(dummy, self._sr, language, beam_size)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            logger.debug("benchmark_sample", i=i, ms=round(elapsed_ms, 1))

        mean_ms = sum(latencies) / len(latencies)
        rtf = (mean_ms / 1000) / audio_duration_s

        result = ASRBenchmarkResult(
            model_size=getattr(self._transcriber, "model_size", "unknown"),
            device=getattr(self._transcriber, "device", "cpu"),
            compute_type=getattr(self._transcriber, "compute_type", "int8"),
            n_samples=n_samples,
            audio_duration_s=audio_duration_s,
            mean_inference_ms=round(mean_ms, 2),
            p50_ms=round(_percentile(latencies, 50), 2),
            p95_ms=round(_percentile(latencies, 95), 2),
            p99_ms=round(_percentile(latencies, 99), 2),
            max_ms=round(max(latencies), 2),
            rtf=round(rtf, 4),
        )

        logger.info(
            "asr_benchmark_complete",
            mean_ms=result.mean_inference_ms,
            p95_ms=result.p95_ms,
            rtf=result.rtf,
            model=result.model_size,
            device=result.device,
        )
        return result

    async def compare(
        self,
        configs: list[dict],
        audio_duration_s: float = 5.0,
        n_samples: int = 10,
    ) -> list[ASRBenchmarkResult]:
        """Run benchmark for multiple language/beam_size configs."""
        results = []
        for cfg in configs:
            r = await self.run(
                audio_duration_s=audio_duration_s,
                n_samples=n_samples,
                language=cfg.get("language", "ru"),
                beam_size=cfg.get("beam_size", 5),
            )
            results.append(r)
        return results
