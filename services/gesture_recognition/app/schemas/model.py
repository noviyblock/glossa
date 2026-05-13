"""Internal Pydantic models for cross-layer data transfer (not API-facing)."""

from __future__ import annotations

from glossa_common.schemas.base import GlossaModel

from ..domain.entities import BenchmarkResult, InferenceDevice, ModelBackend, QuantizationType


def benchmark_to_response(result: BenchmarkResult, mlflow_run_id: str | None = None) -> dict:
    """Convert a domain BenchmarkResult to a serialisable dict for API responses."""
    return {
        "model_id": result.model_id,
        "model_version": result.model_version,
        "backend": result.backend,
        "quantization": result.quantization,
        "n_samples": result.n_samples,
        "batch_size": result.batch_size,
        "latency_mean_ms": round(result.latency.mean_ms, 3),
        "latency_p50_ms": round(result.latency.p50_ms, 3),
        "latency_p95_ms": round(result.latency.p95_ms, 3),
        "latency_p99_ms": round(result.latency.p99_ms, 3),
        "latency_max_ms": round(result.latency.max_ms, 3),
        "throughput_rps": round(result.throughput_rps, 2),
        "warmup_ms": round(result.warmup_ms, 3),
        "mlflow_run_id": mlflow_run_id,
    }
