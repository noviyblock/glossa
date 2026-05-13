"""MLflow helpers for experiment scripts."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


def get_or_create_experiment(name: str) -> str:
    """Return MLflow experiment ID, creating it if needed."""
    try:
        import mlflow  # type: ignore[import]
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            return mlflow.create_experiment(name)
        return exp.experiment_id
    except ImportError:
        return "0"


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that starts an MLflow run or yields a no-op stub."""
    try:
        import mlflow  # type: ignore[import]

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        exp_id = get_or_create_experiment(experiment_name)

        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=tags or {},
        ) as run:
            yield run
    except ImportError:
        # MLflow not installed — yield a stub
        class _NoopRun:
            info = type("_Info", (), {"run_id": "local"})()
        yield _NoopRun()


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow  # type: ignore[import]
        mlflow.log_metrics(metrics)
    except (ImportError, Exception):
        pass


def log_params(params: dict[str, Any]) -> None:
    try:
        import mlflow  # type: ignore[import]
        mlflow.log_params({k: str(v) for k, v in params.items()})
    except (ImportError, Exception):
        pass


def log_artifact(path: str | Path) -> None:
    try:
        import mlflow  # type: ignore[import]
        mlflow.log_artifact(str(path))
    except (ImportError, Exception):
        pass


def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Write results JSON and optionally log to MLflow."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_artifact(output_path)


class Timer:
    """Simple wall-clock timer context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    import statistics
    s = sorted(values)
    idx = max(0, int(p / 100 * len(s)) - 1)
    return s[idx]


def bench_callable(
    fn: Any,
    *args: Any,
    n: int = 50,
    warmup: int = 5,
    **kwargs: Any,
) -> dict[str, float]:
    """Benchmark a callable, return latency statistics in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    timings: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        timings.append((time.perf_counter() - t0) * 1000)

    import statistics
    s = sorted(timings)
    return {
        "mean_ms":   statistics.mean(timings),
        "std_ms":    statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "p50_ms":    percentile(timings, 50),
        "p90_ms":    percentile(timings, 90),
        "p95_ms":    percentile(timings, 95),
        "p99_ms":    percentile(timings, 99),
        "min_ms":    s[0],
        "max_ms":    s[-1],
        "pps":       1000.0 / statistics.mean(timings) if timings else 0.0,
    }
