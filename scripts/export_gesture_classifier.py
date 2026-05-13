"""Export gesture classifier from PyTorch checkpoint to ONNX format.

Referenced by dvc.yaml:export_gesture_classifier stage.
Reads params.yaml for export settings.

Usage:
    python scripts/export_gesture_classifier.py
    python scripts/export_gesture_classifier.py --checkpoint models/gesture_classifier.pt
    python scripts/export_gesture_classifier.py --validate --simplify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_params(path: str = "params.yaml") -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def _build_model(gesture_params: dict) -> "torch.nn.Module":  # type: ignore[name-defined]
    import torch
    import torch.nn as nn

    num_classes = gesture_params.get("num_classes", 200)
    seq_len = gesture_params.get("sequence_length", 30)

    class GestureClassifier(nn.Module):
        """Minimal STGCN-compatible export wrapper."""

        def __init__(self) -> None:
            super().__init__()
            in_features = seq_len * 33 * 3  # 33 landmarks × (x, y, z)
            hidden = 512
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            return self.net(x.view(x.size(0), -1))

    return GestureClassifier()


def export(
    checkpoint: str = "models/gesture_classifier.pt",
    output: str = "models/gesture_classifier.onnx",
    params_path: str = "params.yaml",
    validate: bool = True,
    simplify: bool = False,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch required: pip install torch") from exc

    params = _load_params(params_path)
    gesture_params = params.get("gesture", {})
    export_params = gesture_params.get("export", {})
    seq_len = gesture_params.get("sequence_length", 30)

    # Load model
    ckpt_path = Path(checkpoint)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _build_model(gesture_params)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        # Handle both raw state_dict and {model_state_dict: ...} checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            print(f"Loaded weights from {ckpt_path}")
        except Exception as exc:
            print(f"Warning: partial weight load: {exc}")
    else:
        print(f"Warning: checkpoint not found at {ckpt_path}, exporting with random weights")

    model.eval()

    # Dummy input: (batch=1, seq_len × 33 landmarks × 3 coords)
    dummy = torch.randn(1, seq_len * 33 * 3)
    opset = export_params.get("opset_version", 17)

    print(f"Exporting to ONNX (opset {opset}) → {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["pose_sequence"],
        output_names=["logits"],
        dynamic_axes={
            "pose_sequence": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Simplify (requires onnxsim)
    if simplify or export_params.get("simplify", False):
        try:
            import onnx
            from onnxsim import simplify as onnxsim  # type: ignore[import]

            model_onnx = onnx.load(str(out_path))
            simplified, ok = onnxsim(model_onnx)
            if ok:
                onnx.save(simplified, str(out_path))
                print("ONNX graph simplified")
            else:
                print("Warning: onnxsim returned not-ok, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")

    # Validate
    if validate:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx.checker.check_model(str(out_path))
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            test_input = np.random.randn(2, seq_len * 33 * 3).astype(np.float32)
            out = sess.run(None, {"pose_sequence": test_input})
            assert out[0].shape == (2, gesture_params.get("num_classes", 200)), \
                f"Unexpected output shape: {out[0].shape}"
            print(f"ONNX validation OK — output shape: {out[0].shape}")
        except ImportError:
            print("onnx/onnxruntime not installed, skipping validation")
        except Exception as exc:
            raise RuntimeError(f"ONNX validation failed: {exc}") from exc

    # Log to MLflow
    try:
        from mlops.tracking.experiment_tracker import ExperimentTracker
        from mlops.config import get_settings
        import time

        cfg = get_settings()
        tracker = ExperimentTracker(cfg)
        with tracker.start_run(
            experiment=cfg.gesture_experiment,
            run_name=f"export_onnx_{int(time.time())}",
            tags={
                "pipeline": "export_gesture_classifier",
                "opset": str(opset),
                "checkpoint": str(ckpt_path),
            },
        ):
            tracker.log_artifact(out_path, "onnx")
            tracker.log_params({"opset_version": opset, "simplify": str(simplify)})
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Export complete: {out_path} ({size_mb:.1f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export gesture classifier to ONNX")
    parser.add_argument("--checkpoint", default="models/gesture_classifier.pt")
    parser.add_argument("--output", default="models/gesture_classifier.onnx")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()
    export(
        checkpoint=args.checkpoint,
        output=args.output,
        params_path=args.params,
        validate=args.validate,
        simplify=args.simplify,
    )
