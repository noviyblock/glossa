"""Gesture classifier training pipeline — real ST-GCN on 75 DWPose nodes.

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

# ── Graph topology ────────────────────────────────────────────────────────────
# DWPose (COCO-WholeBody) → 75 nodes, matching the offline dataset-extraction
# script and services/cv_service/keypoint_extractor.py:
#   0-16   COCO body (17 pts)
#   17-22  COCO feet (6 pts)
#   23-32  duplicated key joints (shoulders/hips/knees/ankles), padding to 33
#   33-53  left hand  (21 pts)
#   54-74  right hand (21 pts)

_COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # nose → eyes/ears
    (5, 6), (5, 7), (6, 8), (7, 9),           # shoulders → elbows → wrists
    (5, 11), (6, 12), (11, 12),               # shoulders → hips
    (11, 13), (13, 15), (15, 17), (17, 19),   # left leg
    (12, 14), (14, 16), (16, 18), (18, 20),   # right leg
    (0, 5), (0, 6),                           # nose → shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
]

_FEET_EDGES = [
    (17, 19), (19, 21),  # left foot
    (18, 20), (20, 22),  # right foot
    (17, 18),            # cross-link
]

_EXTRA_EDGES = [
    (23, 5), (24, 6),
    (25, 11), (26, 12),
    (27, 13), (28, 14),
    (29, 15), (30, 16),
    (31, 5), (32, 6),
]

_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17), (5, 17),    # palm cross-links
]

_FACE_NODES = [1, 2, 3, 8, 9, 10]  # connected directly to nose (node 0)

_LEFT_HAND_OFFSET = 33
_RIGHT_HAND_OFFSET = 54
_NUM_NODES = 75


def build_adjacency() -> np.ndarray:
    """Build symmetric-normalised adjacency matrix for 75 DWPose nodes.

    Returns D^{-1/2} (A + I) D^{-1/2} as float32 array of shape (75, 75).
    """
    A = np.eye(_NUM_NODES, dtype=np.float32)
    for i, j in _COCO_EDGES:
        if i < 17 and j < 17:
            A[i, j] = A[j, i] = 1.0
    for i, j in _FEET_EDGES:
        A[i, j] = A[j, i] = 1.0
    for i, j in _EXTRA_EDGES:
        A[i, j] = A[j, i] = 1.0
    for node in _FACE_NODES:
        A[node, 0] = A[0, node] = 1.0
    for i, j in _HAND_EDGES:
        a, b = i + _LEFT_HAND_OFFSET, j + _LEFT_HAND_OFFSET
        A[a, b] = A[b, a] = 1.0
        a, b = i + _RIGHT_HAND_OFFSET, j + _RIGHT_HAND_OFFSET
        A[a, b] = A[b, a] = 1.0
    # Cross: wrists connect to hand root nodes
    A[7, _LEFT_HAND_OFFSET] = A[_LEFT_HAND_OFFSET, 7] = 1.0
    A[4, _RIGHT_HAND_OFFSET] = A[_RIGHT_HAND_OFFSET, 4] = 1.0
    A[5, _LEFT_HAND_OFFSET] = A[_LEFT_HAND_OFFSET, 5] = 1.0
    A[6, _RIGHT_HAND_OFFSET] = A[_RIGHT_HAND_OFFSET, 6] = 1.0
    # Symmetric normalisation
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]).astype(np.float32)


# ── ST-GCN architecture ───────────────────────────────────────────────────────

def _build_stgcn(num_classes: int) -> Any:
    """Return a full STGCN model. Requires PyTorch."""
    import torch
    import torch.nn as nn

    A_np = build_adjacency()
    A_t = torch.tensor(A_np)

    class _GraphConv(nn.Module):
        """Single-partition graph convolution (ONNX-compatible via einsum)."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.register_buffer("A", A_t.clone())
            self.fc = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            self.bn = nn.BatchNorm2d(out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T, V)
            x = self.fc(x)
            x = torch.einsum("bctv,vw->bctw", x, self.A)  # type: ignore[attr-defined]
            return self.bn(x)

    class _TCN(nn.Module):
        def __init__(self, ch: int, stride: int = 1, kernel: int = 9) -> None:
            super().__init__()
            pad = (kernel - 1) // 2
            self.net = nn.Sequential(
                nn.Conv2d(ch, ch, (kernel, 1), stride=(stride, 1), padding=(pad, 0)),
                nn.BatchNorm2d(ch),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _Block(nn.Module):
        def __init__(self, in_ch: int, out_ch: int,
                     stride: int = 1, residual: bool = True) -> None:
            super().__init__()
            self.gcn = _GraphConv(in_ch, out_ch)
            self.tcn = _TCN(out_ch, stride=stride)
            self.relu = nn.ReLU(inplace=True)
            if not residual:
                self.skip: nn.Module = nn.Sequential()  # zero init — handled below
                self._zero_skip = True
            elif in_ch == out_ch and stride == 1:
                self.skip = nn.Identity()
                self._zero_skip = False
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_ch),
                )
                self._zero_skip = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = torch.zeros_like(self.tcn(self.gcn(x))) if self._zero_skip else self.skip(x)
            return self.relu(self.tcn(self.gcn(x)) + res)

    class STGCN(nn.Module):
        """ST-GCN for RSL gesture recognition.

        Input:  (B, T, 75, 3)
        Output: (B, num_classes)
        """

        _LAYER_CFG = [
            # (in_ch, out_ch, stride, residual)
            (3,   64,  1, False),
            (64,  64,  1, True),
            (64,  64,  1, True),
            (64,  64,  1, True),
            (64,  128, 2, True),
            (128, 128, 1, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
            (256, 256, 1, True),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.data_bn = nn.BatchNorm1d(3 * _NUM_NODES)
            self.layers = nn.ModuleList([
                _Block(ic, oc, st, res) for ic, oc, st, res in self._LAYER_CFG
            ])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.drop = nn.Dropout(0.5)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, V, C)
            B, T, V, C = x.shape
            # Apply BN over (B*T, C*V)
            x = x.permute(0, 1, 3, 2).contiguous().view(B * T, C * V)
            x = self.data_bn(x)
            x = x.view(B, T, C, V).permute(0, 2, 1, 3).contiguous()  # (B, C, T, V)
            for layer in self.layers:
                x = layer(x)
            x = self.pool(x).view(B, -1)
            return self.fc(self.drop(x))

    return STGCN()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_dataset(
    data_path: Path,
    split: str = "train",
    slovo_root: str | None = None,
    seq_len: int = 64,
    flip_prob: float = 0.5,
) -> tuple[Any, Any, list[str]]:
    """Load processed gesture dataset.

    Returns X of shape (N, T, 75, 3), y of shape (N,), class_names list.
    Falls back to SlovoDataset when processed data is absent.
    """
    split_dir = data_path / "gestures" / "processed" / split

    if split_dir.exists():
        X = np.load(split_dir / "features.npy")
        y = np.load(split_dir / "labels.npy")
        class_names_path = split_dir.parent / "class_names.json"
        class_names = json.loads(class_names_path.read_text()) if class_names_path.exists() else []
        # Normalise shape from legacy (N, T, 225) to (N, T, 75, 3)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], _NUM_NODES, 3)
        return X.astype(np.float32), y, class_names

    if slovo_root and _SLOVO_AVAILABLE:
        print(f"[train_gesture] Processed split not found — loading from SlovoDataset({slovo_root})")
        ds = SlovoDataset(slovo_root)
        samples = ds.by_split(split)
        class_names = sorted(ds.label_map.keys())

        kps_list: list[np.ndarray] = []
        labels: list[int] = []
        for s in samples:
            try:
                kp = ds.load_keypoints_dwpose(s, target_len=seq_len)
            except Exception:
                continue
            kp = kp.reshape(seq_len, _NUM_NODES, 3)  # ensure (T, 75, 3)
            if split == "train" and np.random.random() < flip_prob:
                kp = horizontal_flip_keypoints(kp)
            kps_list.append(kp)
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


def _export_onnx(model: Any, out_path: Path, params: dict[str, Any]) -> None:
    try:
        import torch
        model.eval()
        seq_len = params.get("sequence_length", 64)
        dummy = torch.randn(1, seq_len, _NUM_NODES, 3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=params.get("export", {}).get("opset_version", 17),
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch_size", 1: "num_frames"},
                "logits":    {0: "batch_size"},
            },
        )
        print(f"Exported ONNX: {out_path}")
    except Exception as exc:
        print(f"ONNX export warning: {exc}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(params_path: str = "params.yaml", dry_run: bool = False,
        slovo_root: str | None = None) -> dict[str, Any]:
    cfg = get_settings()
    params = _load_params(params_path)
    gesture_params = params.get("gesture", {})

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
        "num_nodes": str(_NUM_NODES),
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

        seq_len = gesture_params.get("sequence_length", 64)
        flip_p  = gesture_params.get("augmentation", {}).get("flip_prob", 0.5)
        X_train, y_train, class_names = _load_dataset(
            dataset_path, "train", slovo_root=slovo_root, seq_len=seq_len, flip_prob=flip_p)
        X_val, y_val, _ = _load_dataset(
            dataset_path, "val",   slovo_root=slovo_root, seq_len=seq_len, flip_prob=0.0)

        num_classes = gesture_params.get("num_classes", 1000)
        model = _build_stgcn(num_classes)

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"Training on {device}")

            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.long)
            X_v = torch.tensor(X_val,   dtype=torch.float32).to(device)
            y_v = torch.tensor(y_val,   dtype=torch.long)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=gesture_params.get("learning_rate", 0.001),
                weight_decay=gesture_params.get("weight_decay", 1e-4),
            )
            epochs = gesture_params.get("epochs", 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()
            loader = DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=gesture_params.get("batch_size", 32),
                shuffle=True,
                num_workers=0,
            )

            patience = gesture_params.get("early_stopping_patience", 10)
            best_val_f1 = 0.0
            no_improve = 0
            best_ckpt = cfg.models_path / "gesture_classifier_best.pt"

            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_v)
                    val_pred = val_logits.argmax(dim=1).cpu().numpy().tolist()

                val_metrics = compute_gesture_metrics(y_v.numpy().tolist(), val_pred, class_names)
                tracker.log_metrics({
                    "train_loss":       epoch_loss / len(loader),
                    "val_accuracy":     val_metrics.accuracy,
                    "val_f1_weighted":  val_metrics.f1_weighted,
                    "lr":               scheduler.get_last_lr()[0],
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

            final_ckpt = cfg.models_path / "gesture_classifier.pt"
            torch.save(model.state_dict(), final_ckpt)
            tracker.log_artifact(final_ckpt, "checkpoints")

        except ImportError as exc:
            raise RuntimeError("PyTorch required for training") from exc

        onnx_path = cfg.models_path / "gesture_classifier.onnx"
        _export_onnx(model, onnx_path, gesture_params)
        tracker.log_artifact(onnx_path, "onnx")

        model.eval()
        with torch.no_grad():
            final_pred = model(X_v).argmax(dim=1).cpu().numpy().tolist()
        final_metrics = compute_gesture_metrics(y_v.numpy().tolist(), final_pred, class_names)
        tracker.log_metrics(final_metrics.to_mlflow_dict())
        tracker.set_tag("best_val_f1", f"{best_val_f1:.4f}")

        registered = registry.register_model(
            model_name=cfg.gesture_model_name,
            onnx_path=onnx_path,
            run_id=run_id,
            metrics=final_metrics.to_mlflow_dict(),
            params={k: str(v) for k, v in gesture_params.items()},
            tags=run_tags,
            description=(
                f"ST-GCN gesture classifier — {num_classes} RSL classes, "
                f"{_NUM_NODES} DWPose nodes, seq_len={seq_len}"
            ),
        )

        report = {**final_metrics.to_mlflow_dict(), "run_id": run_id}
        if registered:
            report["registry_version"] = registered.version
        (reports_path / "gesture_train_metrics.json").write_text(json.dumps(report, indent=2))

        print(f"Training complete: accuracy={final_metrics.accuracy:.4f} "
              f"f1={final_metrics.f1_weighted:.4f} run_id={run_id}")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ST-GCN gesture classifier")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--slovo-root", default=None,
                        help="Path to Slovo dataset root (fallback if processed data absent)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(params_path=args.params, dry_run=args.dry_run, slovo_root=args.slovo_root)
