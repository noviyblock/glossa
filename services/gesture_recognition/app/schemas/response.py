"""Pydantic v2 response schemas for the gesture recognition API."""

from __future__ import annotations

from glossa_common.schemas.base import GlossaModel


class GlossCandidateResponse(GlossaModel):
    gloss: str
    confidence: float
    gloss_id: int
    start_frame: int = 0
    end_frame: int = 0


class RecognizeResponse(GlossaModel):
    session_id: str
    glosses: list[str]
    confidences: list[float]
    candidates: list[GlossCandidateResponse] = []
    is_fallback: bool
    inference_ms: float
    queue_wait_ms: float
    model_id: str


class ModelInfoResponse(GlossaModel):
    id: str
    name: str
    architecture: str
    backend: str
    quantization: str
    version: str
    status: str
    is_active: bool
    loaded_at: float | None = None
    description: str = ""


class ModelListResponse(GlossaModel):
    models: list[ModelInfoResponse]
    active_model_id: str | None
    loaded_count: int


class BenchmarkResultResponse(GlossaModel):
    model_id: str
    model_version: str
    backend: str
    quantization: str
    n_samples: int
    batch_size: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    throughput_rps: float
    warmup_ms: float
    mlflow_run_id: str | None = None


class BenchmarkCompareResponse(GlossaModel):
    results: list[BenchmarkResultResponse]
    winner_model_id: str | None = None


class FuzzyMatchResponse(GlossaModel):
    matches: list[tuple[str, float]]


class MessageResponse(GlossaModel):
    message: str
    detail: str | None = None
