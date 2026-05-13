"""Core domain entities — pure Python, zero I/O.

These are the central value objects for the gesture recognition domain.
No external dependencies beyond stdlib and numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np


# ── Enumerations ──────────────────────────────────────────────────────────────

class ModelArchitecture(StrEnum):
    STGCN       = "stgcn"         # Spatial Temporal Graph Conv Network
    POSEFORMER  = "poseformer"    # Transformer on pose sequences
    MLSTM       = "mlstm"         # Multi-layer LSTM (baseline)
    CUSTOM      = "custom"        # Any other ONNX-compatible model


class ModelBackend(StrEnum):
    ONNX        = "onnx"
    OPENVINO    = "openvino"
    # Future-ready
    TENSORRT    = "tensorrt"
    COREML      = "coreml"


class QuantizationType(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8    = "int8"
    INT4    = "int4"    # future


class ModelStatus(StrEnum):
    LOADING      = "loading"
    LOADED       = "loaded"
    HOT_RELOADING = "hot_reloading"
    UNLOADED     = "unloaded"
    FAILED       = "failed"


class InferenceDevice(StrEnum):
    CPU  = "CPU"
    GPU  = "GPU"
    AUTO = "AUTO"   # OpenVINO AUTO plugin


# ── Model specification ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelInputSpec:
    """Describes expected ONNX model input tensor."""
    name: str
    shape: tuple[int, ...]          # use -1 for dynamic axes
    dtype: str = "float32"


@dataclass(frozen=True)
class ModelOutputSpec:
    """Describes expected ONNX model output tensor."""
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass
class ModelSpec:
    """Complete specification of a gesture recognition model."""
    id: str                          # unique slug, e.g. "stgcn-v2-int8"
    name: str                        # human-readable
    architecture: ModelArchitecture
    backend: ModelBackend
    quantization: QuantizationType
    path: Path
    labels_path: Path
    version: str = "1.0.0"
    description: str = ""
    input_spec: ModelInputSpec | None = None
    output_spec: ModelOutputSpec | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Derived at load time
    status: ModelStatus = ModelStatus.UNLOADED
    loaded_at: float | None = None
    load_error: str | None = None

    def mark_loaded(self) -> None:
        self.status = ModelStatus.LOADED
        self.loaded_at = time.time()
        self.load_error = None

    def mark_failed(self, error: str) -> None:
        self.status = ModelStatus.FAILED
        self.load_error = error


# ── Prediction types ──────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class GlossCandidate:
    """Single gloss prediction with confidence."""
    confidence: float               # [0, 1] — sort key
    gloss: str
    gloss_id: int
    start_frame: int = 0
    end_frame: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass
class TopKResult:
    """Top-K gloss candidates for one inference window."""
    candidates: list[GlossCandidate]   # sorted descending by confidence
    is_fallback: bool = False          # True when heuristic-based
    window_start: int = 0
    window_end: int = 0
    raw_logits: np.ndarray | None = field(default=None, repr=False)

    @property
    def best(self) -> GlossCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def best_gloss(self) -> str:
        return self.best.gloss if self.best else "<unknown>"

    @property
    def best_confidence(self) -> float:
        return self.best.confidence if self.best else 0.0


# ── Inference request / result ────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """Single inference request — one sliding window."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    feature_sequence: np.ndarray = field(default_factory=lambda: np.zeros((30, 258), dtype=np.float32))
    top_k: int = 5
    confidence_threshold: float = 0.35
    session_id: str = ""
    submitted_at: float = field(default_factory=time.perf_counter)


@dataclass
class BatchInferenceRequest:
    """Batch of inference requests to run together."""
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    requests: list[InferenceRequest] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.perf_counter)

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    def feature_matrix(self) -> np.ndarray:
        """Stack into [B, T, D] array."""
        return np.stack([r.feature_sequence for r in self.requests], axis=0)


@dataclass
class InferenceResult:
    """Complete result for one InferenceRequest."""
    request_id: str
    session_id: str
    top_k: list[TopKResult]         # one per sliding window position
    model_id: str
    model_version: str
    inference_ms: float
    queue_wait_ms: float
    batch_size: int = 1
    device: InferenceDevice = InferenceDevice.CPU

    @property
    def best_sequence(self) -> list[str]:
        return [r.best_gloss for r in self.top_k if r.best_confidence > 0.35]


# ── Benchmark types ───────────────────────────────────────────────────────────

@dataclass
class LatencyStats:
    """Latency percentile statistics."""
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        arr = np.array(samples_ms, dtype=np.float64)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(np.max(arr)),
        )


@dataclass
class BenchmarkResult:
    """Full benchmark report for one model."""
    model_id: str
    model_version: str
    backend: ModelBackend
    device: InferenceDevice
    quantization: QuantizationType
    n_samples: int
    batch_size: int
    latency: LatencyStats
    throughput_rps: float
    warmup_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "backend": self.backend,
            "device": self.device,
            "quantization": self.quantization,
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "latency_mean_ms": self.latency.mean_ms,
            "latency_p95_ms": self.latency.p95_ms,
            "latency_p99_ms": self.latency.p99_ms,
            "throughput_rps": self.throughput_rps,
            "warmup_ms": self.warmup_ms,
        }


@dataclass
class WarmupResult:
    model_id: str
    iterations: int
    total_ms: float
    mean_ms: float
    success: bool
    error: str | None = None
