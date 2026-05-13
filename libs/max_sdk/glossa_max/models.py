"""Typed models for MAX pipeline requests and responses."""
from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MAXBackend(StrEnum):
    MAX = "max"
    ONNX_FALLBACK = "onnx_fallback"
    UNAVAILABLE = "unavailable"


class MAXModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    dtype: str = "float32"
    batch_size: int = 1


class MAXInferenceRequest(BaseModel):
    session_id: UUID
    model_name: str
    inputs: dict[str, list[float | int]]
    timeout_seconds: float = 30.0


class MAXInferenceResponse(BaseModel):
    session_id: UUID
    model_name: str
    outputs: dict[str, list[float | int]]
    latency_ms: float
    backend: MAXBackend
    tokens_per_second: float | None = None
