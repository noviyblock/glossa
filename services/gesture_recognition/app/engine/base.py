"""Base inference engine — common logic shared by ONNX and OpenVINO backends.

Subclasses must implement:
  _run_inference_sync(batch: np.ndarray) -> np.ndarray

Everything else (async wrapper, retry, top-k, softmax, benchmarking) lives here.
"""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

from ..domain.entities import (
    BenchmarkResult,
    GlossCandidate,
    InferenceDevice,
    LatencyStats,
    ModelBackend,
    ModelSpec,
    TopKResult,
    WarmupResult,
)
from ..domain.interfaces import InferenceEnginePort

logger = get_logger(__name__)

# Maximum logit value before softmax overflow (numerical stability)
_LOGIT_CLIP = 50.0


class BaseInferenceEngine(InferenceEnginePort):
    """Template engine with shared inference utilities.

    Parameters
    ----------
    spec:
        Full model specification.
    labels:
        Ordered list of RSL gloss labels matching model output.
    feature_dim:
        Feature vector dimension per frame.
    executor:
        ThreadPoolExecutor for blocking inference calls.
        If None, asyncio.to_thread is used (default thread pool).
    """

    def __init__(
        self,
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
    ) -> None:
        self._spec = spec
        self._labels = labels
        self._feature_dim = feature_dim
        self._n_threads = n_threads
        self._ready = False
        self._session: object = None   # set by subclass

    # ── InferenceEnginePort ───────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._spec.id

    @property
    def device(self) -> InferenceDevice:
        return InferenceDevice(self._spec.extra.get("device", "CPU"))

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> list[str]:
        return self._labels

    # ── Abstract: subclasses implement the synchronous hot path ───────────────

    @abstractmethod
    def _run_inference_sync(self, batch: np.ndarray) -> np.ndarray:
        """Run synchronous inference.  batch: [B, T, D] → logits: [B, C]."""

    @abstractmethod
    async def _load(self) -> None:
        """Load model from spec.path into self._session."""

    # ── Async wrappers ─────────────────────────────────────────────────────────

    async def predict(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Predict on a single [T, D] sequence.  Returns logits [C]."""
        batch = feature_sequence[np.newaxis, ...]   # [1, T, D]
        logits = await self.predict_batch(batch)
        return logits[0]

    async def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Predict on [B, T, D] batch.  Returns logits [B, C]."""
        if not self._ready:
            raise RuntimeError(f"Engine {self.model_id} is not ready — call _load() first")

        self._validate_batch(batch)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.1, max=1.0),
                retry=retry_if_exception_type(RuntimeError),
                reraise=True,
            ):
                with attempt:
                    with MODEL_INFERENCE_LATENCY.labels(
                        service="gesture-recognition",
                        model=self._spec.architecture,
                    ).time():
                        logits = await asyncio.to_thread(self._run_inference_sync, batch)
        except RetryError:
            raise

        logger.debug(
            "inference_done",
            model_id=self.model_id,
            batch_size=batch.shape[0],
            output_shape=logits.shape,
        )
        return logits

    async def predict_topk(
        self,
        feature_sequence: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.35,
        window_start: int = 0,
        window_end: int = 0,
    ) -> TopKResult:
        """High-level predict: returns top-K GlossCandidates with confidence."""
        logits = await self.predict(feature_sequence)
        return self._logits_to_topk(logits, top_k, threshold, window_start, window_end)

    async def predict_batch_topk(
        self,
        batch: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> list[TopKResult]:
        """Batch top-k prediction."""
        logits_batch = await self.predict_batch(batch)
        return [
            self._logits_to_topk(logits, top_k, threshold)
            for logits in logits_batch
        ]

    # ── Warmup ────────────────────────────────────────────────────────────────

    async def warmup(self, n_iterations: int = 3) -> WarmupResult:
        dummy = np.zeros((1, 30, self._feature_dim), dtype=np.float32)
        t0 = time.perf_counter()
        try:
            for _ in range(n_iterations):
                await self.predict_batch(dummy)
            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "engine_warmup_done",
                model_id=self.model_id,
                iterations=n_iterations,
                total_ms=round(total_ms, 2),
                mean_ms=round(total_ms / n_iterations, 2),
            )
            return WarmupResult(
                model_id=self.model_id,
                iterations=n_iterations,
                total_ms=total_ms,
                mean_ms=total_ms / n_iterations,
                success=True,
            )
        except Exception as exc:
            logger.exception("engine_warmup_failed", model_id=self.model_id)
            return WarmupResult(
                model_id=self.model_id,
                iterations=n_iterations,
                total_ms=0.0,
                mean_ms=0.0,
                success=False,
                error=str(exc),
            )

    # ── Benchmark ─────────────────────────────────────────────────────────────

    async def benchmark(
        self,
        n_samples: int = 100,
        batch_size: int = 1,
        sequence_length: int = 30,
    ) -> BenchmarkResult:
        dummy = np.random.randn(batch_size, sequence_length, self._feature_dim).astype(np.float32)

        # Warmup first
        warmup_result = await self.warmup(n_iterations=5)

        samples_ms: list[float] = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            await self.predict_batch(dummy)
            samples_ms.append((time.perf_counter() - t0) * 1000)

        latency = LatencyStats.from_samples(samples_ms)
        throughput = 1000.0 / latency.mean_ms * batch_size   # samples/sec

        result = BenchmarkResult(
            model_id=self.model_id,
            model_version=self._spec.version,
            backend=self._spec.backend,
            device=self.device,
            quantization=self._spec.quantization,
            n_samples=n_samples,
            batch_size=batch_size,
            latency=latency,
            throughput_rps=throughput,
            warmup_ms=warmup_result.mean_ms,
        )

        logger.info(
            "benchmark_complete",
            model_id=self.model_id,
            n_samples=n_samples,
            batch_size=batch_size,
            mean_ms=round(latency.mean_ms, 2),
            p95_ms=round(latency.p95_ms, 2),
            p99_ms=round(latency.p99_ms, 2),
            throughput_rps=round(throughput, 1),
        )
        return result

    # ── Hot reload ────────────────────────────────────────────────────────────

    async def reload(self, new_spec: ModelSpec) -> None:
        logger.info("engine_hot_reload_start", model_id=self.model_id, new_path=str(new_spec.path))
        old_session = self._session
        self._ready = False
        self._spec = new_spec
        try:
            await self._load()
            logger.info("engine_hot_reload_done", model_id=self.model_id)
        except Exception:
            # Rollback: keep old session
            self._session = old_session
            self._ready = True
            logger.exception("engine_hot_reload_failed_rollback", model_id=self.model_id)
            raise

    # ── Internal utilities ────────────────────────────────────────────────────

    def _logits_to_topk(
        self,
        logits: np.ndarray,
        top_k: int,
        threshold: float,
        window_start: int = 0,
        window_end: int = 0,
    ) -> TopKResult:
        probs = _softmax(np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP))
        k = min(top_k, len(probs))
        top_indices = np.argpartition(probs, -k)[-k:]
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

        candidates = [
            GlossCandidate(
                confidence=float(probs[idx]),
                gloss=self._labels[idx] if idx < len(self._labels) else f"class_{idx}",
                gloss_id=int(idx),
                start_frame=window_start,
                end_frame=window_end,
            )
            for idx in top_indices
            if float(probs[idx]) >= threshold
        ]

        return TopKResult(
            candidates=candidates,
            window_start=window_start,
            window_end=window_end,
            raw_logits=logits.copy(),
        )

    def _validate_batch(self, batch: np.ndarray) -> None:
        if batch.ndim != 3:
            raise ValueError(f"Expected 3D batch [B, T, D], got shape {batch.shape}")
        if batch.shape[2] != self._feature_dim:
            raise ValueError(
                f"Expected feature_dim={self._feature_dim}, got {batch.shape[2]}"
            )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
