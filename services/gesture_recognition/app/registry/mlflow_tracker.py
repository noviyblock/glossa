"""MLflow tracker — benchmark logging and model comparison."""

from __future__ import annotations

import contextlib
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import BenchmarkResult
from ..domain.interfaces import MLflowTrackerPort

logger = get_logger(__name__)


class MLflowTracker(MLflowTrackerPort):
    """Logs benchmark results and model metadata to MLflow.

    Falls back to a no-op if mlflow is not installed or the tracking
    server is unreachable — inference must never fail because of metrics.
    """

    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "gesture-recognition-benchmarks",
    ) -> None:
        self._uri = tracking_uri
        self._experiment = experiment_name
        self._mlflow: Any = None
        self._experiment_id: str | None = None
        self._available = self._try_connect()

    # ── MLflowTrackerPort ─────────────────────────────────────────────────────

    def log_benchmark(self, result: BenchmarkResult) -> str | None:
        """Log a BenchmarkResult as an MLflow run.  Returns run_id or None."""
        if not self._available:
            return None

        try:
            with self._mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=f"{result.model_id}_bench",
            ) as run:
                self._mlflow.set_tags({
                    "model_id": result.model_id,
                    "model_version": result.model_version,
                    "backend": result.backend,
                    "device": result.device,
                    "quantization": result.quantization,
                })
                self._mlflow.log_metrics({
                    "latency_mean_ms": result.latency.mean_ms,
                    "latency_std_ms": result.latency.std_ms,
                    "latency_p50_ms": result.latency.p50_ms,
                    "latency_p90_ms": result.latency.p90_ms,
                    "latency_p95_ms": result.latency.p95_ms,
                    "latency_p99_ms": result.latency.p99_ms,
                    "latency_max_ms": result.latency.max_ms,
                    "throughput_rps": result.throughput_rps,
                    "warmup_ms": result.warmup_ms,
                    "n_samples": result.n_samples,
                    "batch_size": result.batch_size,
                })
                run_id = run.info.run_id
                logger.info(
                    "mlflow_benchmark_logged",
                    model_id=result.model_id,
                    run_id=run_id,
                    p95_ms=round(result.latency.p95_ms, 2),
                )
                return run_id
        except Exception:
            logger.exception("mlflow_log_benchmark_failed", model_id=result.model_id)
            return None

    def compare_models(self, model_ids: list[str]) -> list[dict]:
        """Fetch the latest benchmark run for each model_id and return comparable dicts."""
        if not self._available or not model_ids:
            return []

        try:
            results = []
            for mid in model_ids:
                runs = self._mlflow.search_runs(
                    experiment_ids=[self._experiment_id],
                    filter_string=f"tags.model_id = '{mid}'",
                    order_by=["attributes.start_time DESC"],
                    max_results=1,
                )
                if runs.empty:
                    continue
                row = runs.iloc[0]
                results.append({
                    "model_id": mid,
                    "run_id": row["run_id"],
                    "latency_p95_ms": row.get("metrics.latency_p95_ms"),
                    "throughput_rps": row.get("metrics.throughput_rps"),
                    "backend": row.get("tags.backend"),
                    "quantization": row.get("tags.quantization"),
                })
            return results
        except Exception:
            logger.exception("mlflow_compare_failed")
            return []

    def get_best_model(self, metric: str = "latency_p95_ms", lower_is_better: bool = True) -> str | None:
        """Query MLflow for the model with the best value for `metric`."""
        if not self._available:
            return None

        try:
            order = "ASC" if lower_is_better else "DESC"
            runs = self._mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1,
            )
            if runs.empty:
                return None
            return runs.iloc[0].get("tags.model_id")
        except Exception:
            logger.exception("mlflow_get_best_failed", metric=metric)
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _try_connect(self) -> bool:
        try:
            import mlflow  # type: ignore[import]
            mlflow.set_tracking_uri(self._uri)
            experiment = mlflow.get_experiment_by_name(self._experiment)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(self._experiment)
            else:
                self._experiment_id = experiment.experiment_id
            self._mlflow = mlflow
            logger.info(
                "mlflow_connected",
                uri=self._uri,
                experiment=self._experiment,
                experiment_id=self._experiment_id,
            )
            return True
        except ImportError:
            logger.warning("mlflow_not_installed", hint="pip install mlflow")
            return False
        except Exception:
            logger.warning("mlflow_unreachable", uri=self._uri)
            return False


class NoopMLflowTracker(MLflowTrackerPort):
    """Drop-in replacement when MLflow is disabled."""

    def log_benchmark(self, result: BenchmarkResult) -> str | None:
        return None

    def compare_models(self, model_ids: list[str]) -> list[dict]:
        return []

    def get_best_model(self, metric: str = "latency_p95_ms", lower_is_better: bool = True) -> str | None:
        return None
