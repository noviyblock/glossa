"""Gesture classifier evaluation pipeline.

Usage:
    python -m mlops.pipelines.eval_gesture
    python -m mlops.pipelines.eval_gesture --model-version 3
    python -m mlops.pipelines.eval_gesture --model-path models/gesture_classifier.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlops.config import get_settings
from mlops.registry.onnx_registry import ONNXModelRegistry
from mlops.tracking.experiment_tracker import ExperimentTracker
from mlops.tracking.metrics import compute_gesture_metrics, profile_latency, profile_memory


def _load_onnx_session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install onnxruntime") from exc

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return session


def _run_inference(session: Any, x: np.ndarray, batch_size: int = 32) -> list[int]:
    preds: list[int] = []
    input_name = session.get_inputs()[0].name
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].astype(np.float32)
        logits = session.run(None, {input_name: batch})[0]
        preds.extend(logits.argmax(axis=1).tolist())
    return preds


def _write_confusion_csv(confusion: list[list[int]], class_names: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", *class_names])
        for i, row in enumerate(confusion):
            name = class_names[i] if i < len(class_names) else str(i)
            writer.writerow([name, *row])


def run(
    model_path: str | None = None,
    model_version: str | None = None,
    params_path: str = "params.yaml",
) -> dict[str, Any]:
    cfg = get_settings()
    tracker = ExperimentTracker(cfg)
    registry = ONNXModelRegistry(cfg)

    # Resolve model path
    if model_path:
        onnx_path = Path(model_path)
    else:
        if model_version:
            onnx_path = registry.load_model_path(cfg.gesture_model_name, version=model_version)
        else:
            onnx_path = registry.load_model_path(cfg.gesture_model_name, stage="Staging")
        if onnx_path is None:
            # Fall back to default DVC output
            onnx_path = cfg.models_path / "gesture_classifier.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")

    session = _load_onnx_session(onnx_path)

    # Load test dataset
    # Was "processed"/"test" -- that's the stale T=32/1000-class dataset,
    # and "test" doesn't exist as a split name in the current data at all
    # (only train/val). The actual production model is trained on
    # processed_64_200 (T=64/200-class, see data/gestures/processed_64_200.dvc);
    # "val" is its held-out split, same one used for the TTA/backend A/B
    # comparisons run earlier this session.
    test_dir = cfg.dataset_path / "gestures" / "processed_64_200" / "val"
    if not test_dir.exists():
        raise FileNotFoundError(f"Val split not found: {test_dir}")

    x_test = np.load(test_dir / "features.npy")
    y_test = np.load(test_dir / "labels.npy").tolist()
    class_names_path = test_dir.parent / "class_names.json"
    class_names = (
        json.loads(class_names_path.read_text(encoding="utf-8"))
        if class_names_path.exists()
        else []
    )

    reports_path = cfg.reports_path
    reports_path.mkdir(parents=True, exist_ok=True)

    with tracker.start_run(
        experiment=cfg.gesture_experiment,
        run_name=f"eval_gesture_{int(time.time())}",
        tags={
            "pipeline": "eval_gesture",
            "model_path": str(onnx_path),
        },
        dataset_path=cfg.dataset_path / "gestures",
    ) as run:
        run_id = run.info.run_id

        # Classification metrics
        y_pred = _run_inference(session, x_test)
        metrics = compute_gesture_metrics(y_test, y_pred, class_names)

        # Latency profile
        sample_input = x_test[:1].astype(np.float32)
        input_name = session.get_inputs()[0].name
        latency = profile_latency(
            fn=lambda: session.run(None, {input_name: sample_input}),
            n_samples=cfg.benchmark_n_samples,
            warmup_runs=cfg.benchmark_warmup_runs,
        )

        # Memory profile
        mem_profile, _ = profile_memory(
            lambda: session.run(None, {input_name: x_test[:32].astype(np.float32)})
        )

        # Log everything
        tracker.log_metrics({
            **metrics.to_mlflow_dict(),
            **latency.to_mlflow_dict(),
            **mem_profile.to_mlflow_dict(),
        })

        # Write confusion matrix CSV (DVC plots)
        confusion_path = reports_path / "gesture_confusion_matrix.csv"
        _write_confusion_csv(metrics.confusion_matrix, class_names, confusion_path)
        tracker.log_artifact(confusion_path, "plots")

        # Per-class F1 report
        tracker.log_dict({"per_class_f1": metrics.f1_per_class}, "per_class_f1.json")

        # Compare against current Production if available
        prod = registry.get_production_version(cfg.gesture_model_name)
        if prod:
            prod_lineage = registry.get_lineage(cfg.gesture_model_name, prod.version)
            prod_acc = prod_lineage.get("metrics", {}).get("accuracy", 0.0)
            delta = metrics.accuracy - float(prod_acc)
            tracker.log_metrics({"accuracy_delta_vs_prod": delta})
            tracker.set_tag("compared_vs_prod_version", prod.version)

        # Write DVC metrics file
        report: dict[str, Any] = {
            **metrics.to_mlflow_dict(),
            **latency.to_mlflow_dict(),
            **mem_profile.to_mlflow_dict(),
            "run_id": run_id,
            "n_classes": metrics.n_classes,
            "n_test_samples": metrics.n_samples,
        }
        (reports_path / "gesture_eval_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        _emit_prometheus(metrics, latency, model_version or "unknown")

        print(
            f"Eval complete: accuracy={metrics.accuracy:.4f} "
            f"f1={metrics.f1_weighted:.4f} "
            f"p95={latency.p95_ms:.1f}ms"
        )
        return report


def _emit_prometheus(metrics: Any, latency: Any, version: str) -> None:
    try:
        from glossa_common.telemetry import MODEL_QUALITY  # type: ignore[import]
        for k, v in metrics.to_mlflow_dict().items():
            MODEL_QUALITY.labels(model="gesture", metric=k, version=version, stage="eval").set(v)
        MODEL_QUALITY.labels(
            model="gesture", metric="latency_p95_ms", version=version, stage="eval"
        ).set(latency.p95_ms)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture classifier")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    run(model_path=args.model_path, model_version=args.model_version, params_path=args.params)
