"""MLflow helpers for experiment scripts."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


_DAGSHUB_REPO = "noviyblock/glossa"
_DAGSHUB_MLFLOW_URI = f"https://dagshub.com/{_DAGSHUB_REPO}.mlflow"
_DAGSHUB_S3_ENDPOINT = f"https://dagshub.com/{_DAGSHUB_REPO}.s3"


def setup_kaggle_secrets() -> bool:
    """Load MLflow + DAGsHub DVC credentials from Kaggle Secrets.

    Reads four secrets configured in the Kaggle notebook settings
    (Settings → Add-ons → Secrets):

      MLFLOW_TRACKING_URI      — https://dagshub.com/noviyblock/glossa.mlflow
      MLFLOW_TRACKING_USERNAME — DAGsHub username (noviyblock)
      MLFLOW_TRACKING_PASSWORD — DAGsHub access token
      DAGSHUB_TOKEN            — same DAGsHub token (used for DVC S3 remote)

    The DAGsHub token doubles as both AWS_ACCESS_KEY_ID and
    AWS_SECRET_ACCESS_KEY for the DVC S3 remote on DAGsHub.
    Falls back to MLFLOW_TRACKING_PASSWORD if DAGSHUB_TOKEN is absent.

    Returns True if at least the three MLflow vars were loaded.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import]
        secrets = UserSecretsClient()

        mlflow_keys = {
            "MLFLOW_TRACKING_URI":      "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME": "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD": "MLFLOW_TRACKING_PASSWORD",
        }
        loaded = 0
        for kaggle_key, env_key in mlflow_keys.items():
            try:
                os.environ[env_key] = secrets.get_secret(kaggle_key)
                loaded += 1
            except Exception:
                pass

        # DAGsHub token for DVC S3 remote — same token used as S3 credentials
        token = None
        try:
            token = secrets.get_secret("DAGSHUB_TOKEN")
            os.environ["DAGSHUB_TOKEN"] = token
            loaded += 1
        except Exception:
            token = os.environ.get("MLFLOW_TRACKING_PASSWORD")  # fallback

        if token:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", token)
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", token)

        # DAGsHub S3 endpoint for DVC artifact store
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", _DAGSHUB_S3_ENDPOINT)

        # Default tracking URI to DAGsHub if not set
        os.environ.setdefault("MLFLOW_TRACKING_URI", _DAGSHUB_MLFLOW_URI)

        uri = os.environ.get("MLFLOW_TRACKING_URI", "—")
        if loaded >= 3:
            print(f"[MLflow/DAGsHub] Загружено {loaded}/4 секретов Kaggle.")
            print(f"[MLflow/DAGsHub] Tracking URI: {uri}")
            return True
        print(f"[MLflow/DAGsHub] Загружено {loaded}/4 секретов — "
              f"проверьте Kaggle Secrets (Settings → Add-ons → Secrets).")
        return False
    except ImportError:
        return False  # не на Kaggle — нормальный сценарий


def setup_mlflow(experiment_name: str | None = None) -> None:
    """Configure MLflow via dagshub.init(), with env-var fallback.

    Priority:
      1. dagshub.init() — one-liner that sets tracking URI + auth automatically.
         Uses DAGSHUB_TOKEN env var if present (set by setup_kaggle_secrets()).
      2. Manual URI from MLFLOW_TRACKING_URI env var (local dev fallback).

    Call once at notebook / script top before any mlflow_run() usage.
    """
    # Load Kaggle Secrets first (sets DAGSHUB_TOKEN if on Kaggle)
    setup_kaggle_secrets()

    # Primary: dagshub.init() — handles auth and URI in one call
    try:
        import dagshub  # type: ignore[import]
        dagshub.init(
            repo_owner="noviyblock",
            repo_name="glossa",
            mlflow=True,
        )
        print("[DAGsHub] dagshub.init() OK — "
              f"https://dagshub.com/{_DAGSHUB_REPO}.mlflow")
    except Exception:
        # Fallback: set URI manually (dagshub not installed or no network)
        try:
            import mlflow  # type: ignore[import]
            uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(uri)
            print(f"[MLflow] URI (fallback): {uri}")
        except ImportError:
            pass

    if experiment_name:
        try:
            import mlflow  # type: ignore[import]
            mlflow.set_experiment(experiment_name)
        except ImportError:
            pass


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
    nested: bool = False,
    description: str = "",
) -> Generator[Any, None, None]:
    """Context manager that starts an MLflow run or yields a no-op stub.

    Automatically enriches tags with system info (platform, python_version,
    hostname).  Pass description= to populate the Notes field in the MLflow UI.
    Pass nested=True when calling from a Kaggle notebook that already has
    an active parent run.
    """
    try:
        import mlflow  # type: ignore[import]

        # tracking URI already set by dagshub.init() / setup_mlflow()
        exp_id = get_or_create_experiment(experiment_name)

        auto_tags: dict[str, str] = {
            "platform":       platform.system(),
            "python_version": platform.python_version(),
            "hostname":       socket.gethostname(),
        }
        if description:
            auto_tags["mlflow.note.content"] = description

        all_tags = {**auto_tags, **(tags or {})}

        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=all_tags,
            nested=nested,
        ) as run:
            yield run
    except ImportError:
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


def log_dvc_params(params_path: str | Path = "params.yaml") -> None:
    """Read params.yaml and log scalar values to the active MLflow run.

    Keys are prefixed with 'dvc.<section>.' so they appear grouped in the
    MLflow Params tab.  Skips nested dicts and list values.
    """
    try:
        import yaml  # type: ignore[import]
        import mlflow  # type: ignore[import]

        with open(params_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        flat: dict[str, Any] = {}
        for section, vals in raw.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    if isinstance(v, (int, float, str, bool)) and not isinstance(v, dict):
                        flat[f"dvc.{section}.{k}"] = v
            elif isinstance(vals, (int, float, str, bool)):
                flat[f"dvc.{section}"] = vals

        mlflow.log_params({k: str(v) for k, v in flat.items()})
    except Exception:
        pass


def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Write results JSON and optionally log to MLflow as artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_artifact(output_path)


def load_label_mapping(path_or_uri: str | Path) -> dict:
    """Load label_mapping.json from a local path or an MLflow artifact URI.

    Accepts:
      - Local path:  "data/label_mapping.json"
      - MLflow URI:  "mlflow-artifacts:/RUN_ID/artifacts/label_mapping.json"
      - Runs URI:    "runs:/RUN_ID/label_mapping.json"
    """
    import json as _json

    uri = str(path_or_uri)
    if uri.startswith(("mlflow-artifacts:/", "runs:/")):
        try:
            import mlflow
            local = mlflow.artifacts.download_artifacts(uri)
            with open(local, encoding="utf-8") as f:
                return _json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Cannot download MLflow artifact {uri}: {exc}") from exc

    with open(uri, encoding="utf-8") as f:
        return _json.load(f)


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
