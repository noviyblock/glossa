"""GestureRecognitionSettings — all config via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from glossa_common.config.base import BaseServiceSettings


class GestureRecognitionSettings(BaseServiceSettings):
    model_config = {"env_prefix": "GESTURE_", "case_sensitive": False}

    # Service identity
    service_name: str = "gesture-recognition"
    port: int = 8004

    # Model storage
    models_root: Path = Field(default=Path("/models/gesture"))
    labels_path: Path = Field(default=Path("/models/gesture/labels.txt"))
    frequency_path: Path | None = None

    # Inference
    feature_dim: int = 258
    sequence_length: int = 30
    n_onnx_threads: int | None = None          # None → ORT auto-detect
    warmup_iterations: int = 5
    confidence_threshold: float = 0.35
    top_k: int = 5

    # Sliding window
    window_size: int = 30
    window_stride: int = 15
    max_buffer_frames: int = 120

    # Batch processor
    max_batch_size: int = 8
    batch_timeout_ms: float = 20.0

    # Fallback heuristics
    use_fallback: bool = True
    fallback_confidence_floor: float = 0.15

    # Model file watching
    model_poll_interval_s: float = 30.0

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "gesture-recognition-benchmarks"
    mlflow_enabled: bool = True

    # Auto-load on startup
    autoload_model_id: str | None = None


def get_settings() -> GestureRecognitionSettings:
    return GestureRecognitionSettings()
