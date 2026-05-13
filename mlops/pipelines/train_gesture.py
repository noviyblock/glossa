"""Gesture classifier training pipeline.

Usage:
    python -m mlops.pipelines.train_gesture
    python -m mlops.pipelines.train_gesture --params params.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlops.config import get_settings
from mlops.dataset.versioning import DatasetVersioner
from mlops.registry.onnx_registry import ONNXModelRegistry
from mlops.tracking.experiment_tracker import ExperimentTracker
from mlops.tracking.metrics import compute_gesture_metrics

try:
    from experiments.shared.slovo_dataset import SlovoDataset, horizontal_flip_keypoints
    _SLOVO_AVAILABLE = True
except ImportError:
    _SLOVO_AVAILABLE = False


def _load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
        with open(params_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_model(params: dict[str, Any]) -> Any:
    """Build the gesture classifier model from params.yaml settings."""
    arch = params.get("model_architecture", "stgcn")
    num_classes = params.get("num_classes", 200)
    seq_len = params.get("sequence_length", 30)

    try:
        import torch
        import torch.nn as nn

        if arch == "stgcn":
            # Spatial-Temporal Graph Convolutional Network stub
            # In production, import from services/gesture_recognition
            class _STGCNStub(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = nn.Linear(seq_len * 33 * 3, num_classes)

                def forward(self, x: Any) -> Any:  # type: ignore[override]
                    return self.fc(x.view(x.size(0), -1))

            return _STGCNStub()
        else:
            raise ValueError(f"Unknown architecture: {arch}")
    except ImportError as exc:
        raise RuntimeError("PyTorch required for training: pip install torch") from exc


def _load_dataset(
    data_path: Path,
    split: str = "train",
    slovo_root: str | None = None,
    seq_len: int = 30,
    flip_prob: float = 0.5,
) -> tuple[Any, Any, list[str]]:
    """Load processed gesture dataset.

    Falls back to SlovoDataset when slovo_root is provided and processed
    data is not available yet (typical for first runs on Kaggle/Yandex).
    """
    import numpy as np

    split_dir = data_path / "gestures" / "processed" / split

    if split_dir.exists():
        X = np.load(split_dir / "features.npy")
        y = np.load(split_dir / "labels.npy")
        class_names_path = split_dir.parent / "class_names.json"
        class_names = json.loads(class_names_path.read_text()) if class_names_path.exists() else []
        return X, y, class_names

    if slovo_root and _SLOVO_AVAILABLE:
        print(f"[train_gesture] Processed split not found — loading from SlovoDataset({slovo_root})")
        ds = SlovoDataset(slovo_root)
        samples = ds.by_split(split)
        class_names = sorted(ds.label_map.keys())

        kps_list: list[np.ndarray] = []
        labels: list[int] = []
        for s in samples:
            try:
                kp = ds.load_keypoints_mediapipe(s, target_len=seq_len)
            except Exception:
                continue
            if split == "train" and np.random.random() < flip_prob:
                kp = horizontal_flip_keypoints(kp)
            kps_list.append(kp.reshape(seq_len, -1))
            labels.append(s.label)

        if not kps_list:
            raise RuntimeError(f"No valid samples loaded from SlovoDataset split={split}")

        X = np.stack(kps_list).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y, class_names

    raise FileNotFoundError(
        f"Dataset split not found: {split_dir}\n"
        "Either run: dvc repro preprocess_gesture_dataset\n"
        "Or pass --slovo-root <path> to load directly from Slovo."
    )


def run(params_path: str = "params.yaml", dry_run: bool = False,
        slovo_root: str | None = None) -> dict[str, Any]:
    cfg = get_settings()
    params = _load_params(params_path)
    gesture_params = params.get("gesture", {})
    data_params = params.get("data", {})

    seed = params.get("data", {}).get("random_seed", cfg.random_seed)
    _set_seeds(seed)

    tracker = ExperimentTracker(cfg)
    registry = ONNXModelRegistry(cfg)
    versioner = DatasetVersioner(cfg)
    reports_path = cfg.reports_path
    reports_path.mkdir(parents=True, exist_ok=True)

    dataset_path = cfg.dataset_path
    dataset_info = versioner.snapshot(dataset_path / "gestures")

    run_tags = {
        "pipeline": "train_gesture",
        "architecture": gesture_params.get("model_architecture", "stgcn"),
        "dataset_version": dataset_info.get("dvc_hash", "unknown"),
    }

    with tracker.start_run(
        experiment=cfg.gesture_experiment,
        run_name=f"train_gesture_{int(time.time())}",
        params=gesture_params,
        tags=run_tags,
        dataset_path=dataset_path / "gestures",
    ) as run:
        run_id = run.info.run_id

        if dry_run:
            print(f"[dry-run] Would start training run {run_id}")
            return {"run_id": run_id, "dry_run": True}

        # Load data
        seq_len  = gesture_params.get("sequence_length", 30)
        flip_p   = gesture_params.get("augmentation", {}).get("flip_prob", 0.5)
        X_train, y_train, class_names = _load_dataset(
            dataset_path, "train", slovo_root=slovo_root, seq_len=seq_len, flip_prob=flip_p)
        X_val, y_val, _ = _load_dataset(
            dataset_path, "val", slovo_root=slovo_root, seq_len=seq_len, flip_prob=0.0)

        model = _build_model(gesture_params)

        # Training loop
        epochs = gesture_params.get("epochs", 100)
        lr = gesture_params.get("learning_rate", 0.001)
        patience = gesture_params.get("early_stopping_patience", 10)
        best_val_f1 = 0.0
        no_improve = 0
        best_ckpt = cfg.models_path / "gesture_classifier_best.pt"

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader

            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.long)
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.long)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr,
                weight_decay=gesture_params.get("weight_decay", 1e-4),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()
            loader = DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=gesture_params.get("batch_size", 32),
                shuffle=True,
            )

            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                scheduler.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_v)
                    val_pred = val_logits.argmax(dim=1).numpy().tolist()

                val_metrics = compute_gesture_metrics(y_v.numpy().tolist(), val_pred, class_names)
                tracker.log_metrics({
                    "train_loss": epoch_loss / len(loader),
                    "val_accuracy": val_metrics.accuracy,
                    "val_f1_weighted": val_metrics.f1_weighted,
                    "lr": scheduler.get_last_lr()[0],
                }, step=epoch)

                if val_metrics.f1_weighted > best_val_f1:
                    best_val_f1 = val_metrics.f1_weighted
                    no_improve = 0
                    torch.save(model.state_dict(), best_ckpt)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            # Save final checkpoint
            final_ckpt = cfg.models_path / "gesture_classifier.pt"
            torch.save(model.state_dict(), final_ckpt)
            tracker.log_artifact(final_ckpt, "checkpoints")

        except ImportError as exc:
            raise RuntimeError("PyTorch required for training") from exc

        # Export to ONNX (triggers eval pipeline via dvc repro)
        onnx_path = cfg.models_path / "gesture_classifier.onnx"
        _export_onnx(model, onnx_path, gesture_params)
        tracker.log_artifact(onnx_path, "onnx")

        # Final validation metrics
        model.eval()
        with torch.no_grad():
            final_pred = model(X_v).argmax(dim=1).numpy().tolist()
        final_metrics = compute_gesture_metrics(y_v.numpy().tolist(), final_pred, class_names)
        tracker.log_metrics(final_metrics.to_mlflow_dict())
        tracker.set_tag("best_val_f1", f"{best_val_f1:.4f}")

        # Register in MLflow Model Registry
        registered = registry.register_model(
            model_name=cfg.gesture_model_name,
            onnx_path=onnx_path,
            run_id=run_id,
            metrics=final_metrics.to_mlflow_dict(),
            params={k: str(v) for k, v in gesture_params.items()},
            tags=run_tags,
            description=f"STGCN gesture classifier, {final_metrics.n_classes} classes",
        )

        # Write DVC metrics report
        report = {**final_metrics.to_mlflow_dict(), "run_id": run_id}
        if registered:
            report["registry_version"] = registered.version
        (reports_path / "gesture_train_metrics.json").write_text(json.dumps(report, indent=2))

        print(f"Training complete: accuracy={final_metrics.accuracy:.4f} "
              f"f1={final_metrics.f1_weighted:.4f} "
              f"run_id={run_id}")
        return report


def _export_onnx(model: Any, out_path: Path, params: dict[str, Any]) -> None:
    try:
        import torch
        model.eval()
        seq_len = params.get("sequence_length", 30)
        dummy = torch.randn(1, seq_len * 33 * 3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=params.get("export", {}).get("opset_version", 17),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"Exported ONNX: {out_path}")
    except Exception as exc:
        print(f"ONNX export warning: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--slovo-root", default=None,
                        help="Path to Slovo dataset root (fallback if processed data absent)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(params_path=args.params, dry_run=args.dry_run, slovo_root=args.slovo_root)
