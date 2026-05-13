"""Pydantic v2 request schemas for the gesture recognition API."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator

from glossa_common.schemas.base import GlossaModel

from ..domain.entities import ModelArchitecture, ModelBackend, QuantizationType


class RecognizeRequest(GlossaModel):
    """Single-shot gesture recognition from a pre-extracted feature sequence."""

    session_id: str = Field(default="", description="Client session identifier")
    feature_sequence: list[list[float]] = Field(
        ...,
        description="Feature matrix [T, D] — T frames × D features (default D=258)",
        min_length=1,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @field_validator("feature_sequence")
    @classmethod
    def validate_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if not v or not v[0]:
            raise ValueError("feature_sequence must be non-empty")
        dim = len(v[0])
        if not all(len(row) == dim for row in v):
            raise ValueError("All rows in feature_sequence must have the same length")
        return v


class LoadModelRequest(GlossaModel):
    """Request to load a new model into the registry."""

    model_id: str = Field(..., description="Unique slug for the model")
    name: str = Field(default="", description="Human-readable name")
    architecture: ModelArchitecture = ModelArchitecture.STGCN
    backend: ModelBackend = ModelBackend.ONNX
    quantization: QuantizationType = QuantizationType.FLOAT32
    path: str = Field(..., description="Path to the ONNX model file")
    labels_path: str = Field(..., description="Path to the labels/glosses file")
    version: str = "1.0.0"
    description: str = ""
    set_active: bool = Field(default=False, description="Make this model active immediately")


class HotReloadRequest(GlossaModel):
    """Request to atomically replace a running model."""

    new_path: str = Field(..., description="Path to the new ONNX model file")
    new_version: str = Field(..., description="Version string for the new model")
    labels_path: str | None = None


class BenchmarkRequest(GlossaModel):
    """Request to run a latency benchmark on one or more models."""

    model_ids: list[str] = Field(..., min_length=1, description="Model IDs to benchmark")
    n_samples: int = Field(default=100, ge=10, le=10000)
    batch_size: int = Field(default=1, ge=1, le=64)
    warmup_iters: int = Field(default=10, ge=1, le=100)


class SetActiveRequest(GlossaModel):
    model_id: str


class FuzzyMatchRequest(GlossaModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
