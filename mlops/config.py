"""MLOps configuration — MLflow, DVC, registry, and pipeline settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLOPS_",
        env_file=".env",
        extra="ignore",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_registry_uri: str | None = None  # defaults to tracking_uri
    mlflow_artifact_root: str = "/mlartifacts"

    # Experiment names
    gesture_experiment: str = "glossa-gesture-recognition"
    nlp_experiment: str = "glossa-nlp-translation"
    asr_experiment: str = "glossa-asr"
    benchmark_experiment: str = "glossa-benchmarks"

    # MLflow Model Registry names
    gesture_model_name: str = "glossa-gesture-classifier"
    nlp_model_name: str = "glossa-nlp-translator"
    asr_model_name: str = "glossa-asr-whisper"

    # ── DVC ───────────────────────────────────────────────────────────────────
    dvc_remote: str = "s3"
    dataset_path: Path = Path("data")
    models_path: Path = Path("models")
    reports_path: Path = Path("mlops/reports")

    # ── Promotion thresholds ──────────────────────────────────────────────────
    gesture_min_accuracy: float = Field(default=0.90, ge=0.0, le=1.0)
    gesture_min_f1: float = Field(default=0.88, ge=0.0, le=1.0)
    gesture_max_latency_p95_ms: float = Field(default=50.0, gt=0.0)

    nlp_min_bleu: float = Field(default=0.35, ge=0.0, le=1.0)
    nlp_min_rouge_l: float = Field(default=0.40, ge=0.0, le=1.0)
    nlp_max_latency_p95_ms: float = Field(default=500.0, gt=0.0)

    asr_max_wer: float = Field(default=0.15, ge=0.0, le=1.0)

    # ── Benchmarking ──────────────────────────────────────────────────────────
    benchmark_n_samples: int = Field(default=100, ge=1)
    benchmark_warmup_runs: int = Field(default=10, ge=0)
    benchmark_batch_sizes: list[int] = Field(default=[1, 4, 8])
    benchmark_backends: list[str] = Field(default=["onnx"])

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42


_settings: MLOpsSettings | None = None


def get_settings() -> MLOpsSettings:
    global _settings
    if _settings is None:
        _settings = MLOpsSettings()
    return _settings
