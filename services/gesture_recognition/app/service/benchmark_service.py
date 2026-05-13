"""Benchmark service — A/B comparison, latency profiling, MLflow logging."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import (
    BenchmarkResult,
    InferenceDevice,
    LatencyStats,
    ModelSpec,
)
from ..domain.interfaces import InferenceEnginePort, MLflowTrackerPort, ModelRegistryPort

logger = get_logger(__name__)


class BenchmarkService:
    """Run latency benchmarks on loaded inference engines.

    Usage::

        result = await svc.run(model_id="stgcn-v2", n_samples=200, batch_size=1)
        comparison = await svc.compare(["stgcn-v2", "poseformer-v1"])
    """

    def __init__(
        self,
        registry: ModelRegistryPort,
        mlflow_tracker: MLflowTrackerPort,
        feature_dim: int = 258,
        sequence_length: int = 30,
    ) -> None:
        self._registry = registry
        self._tracker = mlflow_tracker
        self._feature_dim = feature_dim
        self._seq_len = sequence_length

    async def run(
        self,
        model_id: str,
        n_samples: int = 100,
        batch_size: int = 1,
        warmup_iters: int = 10,
        device: InferenceDevice = InferenceDevice.CPU,
    ) -> BenchmarkResult:
        """Benchmark a single model. Logs results to MLflow."""
        engine = self._registry.get(model_id)
        spec = self._get_spec(model_id)

        logger.info(
            "benchmark_start",
            model_id=model_id,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        # Warmup
        warmup_result = await engine.warmup(warmup_iters)
        if not warmup_result.success:
            raise RuntimeError(f"Warmup failed for {model_id}: {warmup_result.error}")

        # Generate synthetic inputs
        dummy = self._make_dummy(batch_size)
        latencies: list[float] = []

        for _ in range(n_samples):
            t0 = time.perf_counter()
            await engine.infer(dummy[0] if batch_size == 1 else dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        stats = LatencyStats.from_samples(latencies)
        throughput = (n_samples * batch_size) / (sum(latencies) / 1000.0)

        result = BenchmarkResult(
            model_id=model_id,
            model_version=spec.version if spec else "unknown",
            backend=spec.backend if spec else "onnx",
            device=device,
            quantization=spec.quantization if spec else "float32",
            n_samples=n_samples,
            batch_size=batch_size,
            latency=stats,
            throughput_rps=throughput,
            warmup_ms=warmup_result.mean_ms,
        )

        run_id = self._tracker.log_benchmark(result)
        logger.info(
            "benchmark_complete",
            model_id=model_id,
            p95_ms=round(stats.p95_ms, 2),
            p99_ms=round(stats.p99_ms, 2),
            throughput_rps=round(throughput, 1),
            mlflow_run_id=run_id,
        )
        return result

    async def compare(
        self,
        model_ids: list[str],
        n_samples: int = 100,
        batch_size: int = 1,
    ) -> list[BenchmarkResult]:
        """Run benchmarks on multiple models concurrently and return results."""
        tasks = [
            self.run(mid, n_samples=n_samples, batch_size=batch_size)
            for mid in model_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[BenchmarkResult] = []
        for mid, res in zip(model_ids, results):
            if isinstance(res, Exception):
                logger.error("benchmark_compare_error", model_id=mid, error=str(res))
            else:
                out.append(res)

        if len(out) >= 2:
            out.sort(key=lambda r: r.latency.p95_ms)
            winner = out[0]
            logger.info(
                "benchmark_winner",
                model_id=winner.model_id,
                p95_ms=round(winner.latency.p95_ms, 2),
            )

        return out

    def get_mlflow_comparison(self, model_ids: list[str]) -> list[dict]:
        return self._tracker.compare_models(model_ids)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_dummy(self, batch_size: int) -> np.ndarray:
        return np.random.randn(batch_size, self._seq_len, self._feature_dim).astype(np.float32)

    def _get_spec(self, model_id: str) -> ModelSpec | None:
        for spec in self._registry.list_all():
            if spec.id == model_id:
                return spec
        return None
