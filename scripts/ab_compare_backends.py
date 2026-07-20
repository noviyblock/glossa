"""A/B comparison: OpenVINO INT8 (production) vs full-precision ONNX
(gesture_classifier.onnx) on the same inputs.

Quantization to INT8 sometimes costs a bit of accuracy for a lot of speed —
this script measures whether that trade is actually paying off on real data,
rather than assuming it.

Usage (from repo root):
    python scripts/ab_compare_backends.py \
        --features data/gestures/processed_64_200/val/features.npy \
        --labels   data/gestures/processed_64_200/val/labels.npy \
        --n 200

Or against raw (un-normalized) live-extracted clips, one .npy per file,
shape (T, 75, 3), T resampled to WINDOW_SIZE beforehand:
    python scripts/ab_compare_backends.py --clips-dir path/to/rtmw_clips \
        --labels-csv path/to/labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "cv_service"))


def load_openvino(xml_path: str):
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    core = Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    output = next(iter(compiled.outputs))

    def predict(window: np.ndarray) -> np.ndarray:
        inp = window[np.newaxis].astype(np.float32)
        return compiled([inp])[output][0]

    return predict


def load_onnx(onnx_path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def predict(window: np.ndarray) -> np.ndarray:
        inp = window[np.newaxis].astype(np.float32)
        return sess.run(None, {input_name: inp})[0][0]

    return predict


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/gestures/processed_64_200/val/features.npy")
    ap.add_argument("--labels",   default="data/gestures/processed_64_200/val/labels.npy")
    ap.add_argument("--norm-stats", default="models/norm_stats.npz")
    ap.add_argument("--ov-xml",   default="models/stgcn_topk_int8/stgcn_topk_int8.xml")
    ap.add_argument("--onnx",     default="models/gesture_classifier.onnx")
    ap.add_argument("--class-map", default="data/idx_to_class.json")
    ap.add_argument("--n", type=int, default=200, help="max samples to evaluate (0 = all)")
    args = ap.parse_args()

    print(f"Loading OpenVINO INT8 from {args.ov_xml} ...")
    ov_predict = load_openvino(args.ov_xml)
    print(f"Loading full-precision ONNX from {args.onnx} ...")
    onnx_predict = load_onnx(args.onnx)

    idx_to_class = {int(k): v for k, v in json.loads(Path(args.class_map).read_text(encoding="utf-8")).items()}

    X = np.load(args.features, mmap_mode="r")
    y = np.load(args.labels)
    stats = np.load(args.norm_stats)
    mean, std = stats["mean"][0], stats["std"][0]
    std = np.where(std < 1e-6, 1.0, std)

    n = len(X) if args.n <= 0 else min(args.n, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))

    ov_correct = onnx_correct = agree = 0
    ov_conf_sum = onnx_conf_sum = 0.0
    ov_time = onnx_time = 0.0
    disagreements: list[tuple[int, str, str, str]] = []

    for i in idx:
        window = ((np.array(X[i]).astype(np.float32) - mean) / std).astype(np.float32)
        true_label = int(y[i])

        t0 = time.perf_counter()
        ov_probs = softmax(ov_predict(window))
        ov_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        onnx_probs = softmax(onnx_predict(window))
        onnx_time += time.perf_counter() - t0

        ov_top1, onnx_top1 = int(ov_probs.argmax()), int(onnx_probs.argmax())
        ov_correct += ov_top1 == true_label
        onnx_correct += onnx_top1 == true_label
        agree += ov_top1 == onnx_top1
        ov_conf_sum += ov_probs[ov_top1]
        onnx_conf_sum += onnx_probs[onnx_top1]

        if ov_top1 != onnx_top1 and len(disagreements) < 20:
            disagreements.append((
                int(i),
                idx_to_class.get(true_label, str(true_label)),
                idx_to_class.get(ov_top1, str(ov_top1)),
                idx_to_class.get(onnx_top1, str(onnx_top1)),
            ))

    print(f"\n=== A/B: OpenVINO INT8 vs full ONNX ({n} samples) ===")
    print(f"{'':20s} {'accuracy':>10s} {'mean conf':>10s} {'avg latency':>14s}")
    print(f"{'OpenVINO INT8':20s} {ov_correct/n:>10.4f} {ov_conf_sum/n:>10.4f} {1000*ov_time/n:>12.2f}ms")
    print(f"{'Full ONNX':20s} {onnx_correct/n:>10.4f} {onnx_conf_sum/n:>10.4f} {1000*onnx_time/n:>12.2f}ms")
    print(f"\nTop-1 agreement between backends: {agree/n:.4f}")

    if disagreements:
        print(f"\nSample disagreements (true / OpenVINO pred / ONNX pred), showing up to 20:")
        for i, true_c, ov_c, onnx_c in disagreements:
            print(f"  idx={i:5d}  true={true_c!r:20s}  ov={ov_c!r:20s}  onnx={onnx_c!r:20s}")

    acc_diff = onnx_correct - ov_correct
    if acc_diff == 0:
        verdict = "No accuracy difference — INT8 quantization is free here, keep it for the speed."
    elif acc_diff > 0:
        verdict = (f"Full ONNX is {acc_diff/n*100:.1f}pp more accurate — INT8 IS costing accuracy. "
                   f"Worth it only if the {1000*(onnx_time-ov_time)/n:.1f}ms/frame speed difference matters "
                   f"for your latency budget.")
    else:
        verdict = "OpenVINO INT8 is (slightly) MORE accurate than full ONNX on this sample — noise, not a real effect; re-run with more samples."
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
