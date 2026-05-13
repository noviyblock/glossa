"""Unified MLflow experiment tracker with dataset lineage and run linking."""

from __future__ import annotations

import contextlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from mlops.config import MLOpsSettings, get_settings


class ExperimentTracker:
    """Wraps MLflow with Glossa-specific conventions.

    - Auto-creates experiments on first use
    - Tags every run with dataset DVC hash for reproducibility
    - Supports parent/child run linking (train → eval)
    - Falls back to no-op when MLflow is unreachable
    """

    def __init__(self, settings: MLOpsSettings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._mlflow: Any = None
        self._experiment_ids: dict[str, str] = {}
        self._available = self._connect()

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def start_run(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        parent_run_id: str | None = None,
        dataset_path: str | Path | None = None,
    ) -> Generator[Any, None, None]:
        """Context manager yielding an active MLflow run (or a no-op stub)."""
        if not self._available:
            yield _NoopRun()
            return

        exp_id = self._get_or_create_experiment(experiment)
        all_tags: dict[str, str] = {}
        if dataset_path:
            dvc_hash = _get_dvc_hash(dataset_path)
            if dvc_hash:
                all_tags["dataset_dvc_hash"] = dvc_hash
                all_tags["dataset_path"] = str(dataset_path)
        if tags:
            all_tags.update(tags)

        kwargs: dict[str, Any] = {
            "experiment_id": exp_id,
            "run_name": run_name,
            "tags": all_tags or None,
        }
        if parent_run_id:
            kwargs["parent_run_id"] = parent_run_id

        with self._mlflow.start_run(**kwargs) as run:
            if params:
                self._mlflow.log_params(params)
            yield run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._available:
            self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._available:
            # MLflow requires string values for params
            self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        if self._available:
            self._mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        if self._available:
            self._mlflow.log_dict(data, filename)

    def set_tag(self, key: str, value: str) -> None:
        if self._available:
            self._mlflow.set_tag(key, value)

    def search_runs(
        self,
        experiment: str,
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Return simplified run dicts from an experiment query."""
        if not self._available:
            return []
        try:
            exp_id = self._get_or_create_experiment(experiment)
            df = self._mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string=filter_string,
                order_by=order_by or ["attributes.start_time DESC"],
                max_results=max_results,
            )
            return df.to_dict(orient="records") if not df.empty else []
        except Exception:
            return []

    def get_best_run(
        self,
        experiment: str,
        metric: str,
        lower_is_better: bool = False,
    ) -> dict[str, Any] | None:
        order = "ASC" if lower_is_better else "DESC"
        runs = self.search_runs(
            experiment,
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        return runs[0] if runs else None

    @property
    def available(self) -> bool:
        return self._available

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            import mlflow  # type: ignore[import]
            mlflow.set_tracking_uri(self._cfg.mlflow_tracking_uri)
            if self._cfg.mlflow_registry_uri:
                mlflow.set_registry_uri(self._cfg.mlflow_registry_uri)
            self._mlflow = mlflow
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _get_or_create_experiment(self, name: str) -> str:
        if name in self._experiment_ids:
            return self._experiment_ids[name]
        exp = self._mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = self._mlflow.create_experiment(
                name,
                artifact_location=f"{self._cfg.mlflow_artifact_root}/{name}",
            )
        else:
            exp_id = exp.experiment_id
        self._experiment_ids[name] = exp_id
        return exp_id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_dvc_hash(path: str | Path) -> str | None:
    """Return DVC content hash for a tracked path, or None if not tracked."""
    dvc_file = Path(str(path) + ".dvc")
    if dvc_file.exists():
        try:
            import yaml  # type: ignore[import]
            data = yaml.safe_load(dvc_file.read_text())
            outs = data.get("outs", [])
            if outs:
                return outs[0].get("md5") or outs[0].get("hash")
        except Exception:
            pass
    # Try reading from dvc lock
    lock_file = Path("dvc.lock")
    if lock_file.exists():
        try:
            import yaml  # type: ignore[import]
            lock = yaml.safe_load(lock_file.read_text())
            for stage in lock.get("stages", {}).values():
                for out in stage.get("outs", []):
                    if Path(out.get("path", "")) == Path(path):
                        return out.get("md5") or out.get("hash")
        except Exception:
            pass
    return None


class _NoopRun:
    """Stub run returned when MLflow is unavailable."""
    class info:
        run_id = "noop"
    def __enter__(self) -> _NoopRun:
        return self
    def __exit__(self, *_: Any) -> None:
        pass
