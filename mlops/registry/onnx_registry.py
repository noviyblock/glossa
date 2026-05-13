"""ONNX Model Registry backed by MLflow Model Registry."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlops.config import MLOpsSettings, get_settings


@dataclass
class RegisteredModel:
    name: str
    version: str
    stage: str          # None | Staging | Production | Archived
    run_id: str
    artifact_uri: str
    onnx_path: str | None
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.stage == "Production"

    @property
    def is_staging(self) -> bool:
        return self.stage == "Staging"


class ONNXModelRegistry:
    """MLflow-backed registry for ONNX models with lineage tracking.

    Provides:
    - register_model(): store ONNX + metadata in MLflow Model Registry
    - load_model(): retrieve ONNX bytes by name/version/stage
    - list_versions(): all registered versions for a model
    - compare_versions(): side-by-side metric comparison
    - get_lineage(): full provenance chain for a version
    """

    def __init__(self, settings: MLOpsSettings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._mlflow: Any = None
        self._client: Any = None
        self._available = self._connect()

    # ── Registration ──────────────────────────────────────────────────────────

    def register_model(
        self,
        model_name: str,
        onnx_path: str | Path,
        run_id: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
    ) -> RegisteredModel | None:
        """Register an ONNX file as a new version in the MLflow Model Registry."""
        if not self._available:
            return None

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Compute SHA-256 for integrity check
        sha256 = _file_sha256(onnx_path)

        try:
            with self._mlflow.start_run(run_id=run_id):
                # Log ONNX as artifact
                self._mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

                # Log metrics and params if provided
                if metrics:
                    self._mlflow.log_metrics(metrics)
                if params:
                    self._mlflow.log_params({k: str(v) for k, v in params.items()})

                # Tag with provenance info
                all_tags = {
                    "onnx_sha256": sha256,
                    "onnx_filename": onnx_path.name,
                    "model_type": "onnx",
                }
                if tags:
                    all_tags.update(tags)
                self._mlflow.set_tags(all_tags)

            # Register in Model Registry
            artifact_uri = f"runs:/{run_id}/onnx/{onnx_path.name}"
            mv = self._mlflow.register_model(artifact_uri, model_name)

            # Apply extra tags to the model version
            if tags:
                for k, v in tags.items():
                    self._client.set_model_version_tag(model_name, mv.version, k, v)
            self._client.set_model_version_tag(model_name, mv.version, "onnx_sha256", sha256)

            if description:
                self._client.update_model_version(
                    name=model_name,
                    version=mv.version,
                    description=description,
                )

            return RegisteredModel(
                name=model_name,
                version=mv.version,
                stage=mv.current_stage or "None",
                run_id=run_id,
                artifact_uri=artifact_uri,
                onnx_path=str(onnx_path),
                tags=all_tags,
                metrics=metrics or {},
                params=params or {},
            )

        except Exception as exc:
            raise RuntimeError(f"Failed to register model {model_name}") from exc

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_model_path(
        self,
        model_name: str,
        version: str | None = None,
        stage: str = "Production",
    ) -> Path | None:
        """Download and return local path to the ONNX artifact."""
        if not self._available:
            return None

        try:
            if version:
                uri = f"models:/{model_name}/{version}"
            else:
                uri = f"models:/{model_name}/{stage}"

            local_dir = self._mlflow.artifacts.download_artifacts(uri)
            # Find the .onnx file inside the downloaded directory
            onnx_files = list(Path(local_dir).rglob("*.onnx"))
            return onnx_files[0] if onnx_files else None
        except Exception:
            return None

    # ── Listing & comparison ──────────────────────────────────────────────────

    def list_versions(self, model_name: str) -> list[RegisteredModel]:
        if not self._available:
            return []
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            return [self._mv_to_registered(mv) for mv in versions]
        except Exception:
            return []

    def compare_versions(
        self,
        model_name: str,
        metric_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a comparison table of all versions sorted by creation time."""
        if not self._available:
            return []

        default_keys = [
            "accuracy", "f1_weighted", "latency_p95_ms",
            "throughput_rps", "bleu_4", "wer",
        ]
        keys = metric_keys or default_keys

        rows = []
        for rv in self.list_versions(model_name):
            run_data = self._get_run_data(rv.run_id)
            row: dict[str, Any] = {
                "version": rv.version,
                "stage": rv.stage,
                "run_id": rv.run_id,
            }
            for k in keys:
                row[k] = run_data.get(f"metrics.{k}", run_data.get(k))
            row.update({f"tag_{k}": v for k, v in rv.tags.items()})
            rows.append(row)

        return rows

    def get_lineage(self, model_name: str, version: str) -> dict[str, Any]:
        """Return full provenance: run params, dataset hash, parent run."""
        if not self._available:
            return {}
        try:
            mv = self._client.get_model_version(model_name, version)
            run_data = self._get_run_data(mv.run_id)
            return {
                "model_name": model_name,
                "version": version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "params": {
                    k.replace("params.", ""): v
                    for k, v in run_data.items()
                    if k.startswith("params.")
                },
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in run_data.items()
                    if k.startswith("metrics.")
                },
                "tags": {
                    k.replace("tags.", ""): v
                    for k, v in run_data.items()
                    if k.startswith("tags.")
                },
                "dataset_dvc_hash": run_data.get("tags.dataset_dvc_hash"),
                "parent_run_id": run_data.get("tags.mlflow.parentRunId"),
            }
        except Exception:
            return {}

    def get_production_version(self, model_name: str) -> RegisteredModel | None:
        versions = [v for v in self.list_versions(model_name) if v.is_production]
        return versions[0] if versions else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            import mlflow  # type: ignore[import]
            from mlflow.tracking import MlflowClient  # type: ignore[import]
            mlflow.set_tracking_uri(self._cfg.mlflow_tracking_uri)
            self._mlflow = mlflow
            self._client = MlflowClient(self._cfg.mlflow_tracking_uri)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _mv_to_registered(self, mv: Any) -> RegisteredModel:
        return RegisteredModel(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage or "None",
            run_id=mv.run_id,
            artifact_uri=mv.source,
            onnx_path=None,
            tags={t.key: t.value for t in (mv.tags or [])},
        )

    def _get_run_data(self, run_id: str) -> dict[str, Any]:
        try:
            run = self._client.get_run(run_id)
            data = {}
            data.update({f"metrics.{k}": v for k, v in run.data.metrics.items()})
            data.update({f"params.{k}": v for k, v in run.data.params.items()})
            data.update({f"tags.{k}": v for k, v in run.data.tags.items()})
            return data
        except Exception:
            return {}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
