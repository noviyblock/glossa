"""Experiment 02 — Sliding Window Parameter Search.

Grid search over (window_size × stride) combinations to find the optimal
balance between gesture recognition accuracy, latency, and PPS.

Reproduces and extends the analysis from:
  Sber AI (2023) — "Распознавание РЖЯ: 3+ жестов в секунду без GPU"

Grid:
  window_size ∈ {16, 30, 48, 64}  frames
  stride      ∈ { 1,  8, 15, 30}  frames (overlap = window - stride)

Key finding expected:
  - stride=1  → best accuracy, worst PPS (high overlap)
  - stride=15 → good accuracy, ~real-time on CPU  ← our production choice
  - stride=30 → fastest, accuracy degrades notably

Usage:
    python -m experiments.02_sliding_window.run \
        --slovo-root data/slovo \
        --onnx-model models/gesture_classifier.onnx
    python -m experiments.02_sliding_window.run --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.shared.mlflow_utils import (
    log_metrics, log_params, mlflow_run, save_results,
)

EXPERIMENT_NAME = "02_sliding_window_search"
RESULTS_DIR = Path("experiments/results/02_sliding_window")

WINDOW_SIZES = [16, 30, 48, 64]
STRIDES      = [1, 8, 15, 30]


# ── Sliding window inference over a sequence ──────────────────────────────────

def sliding_window_predict(
    session: Any,
    keypoints: np.ndarray,   # (T_full, 75, 3)
    window: int,
    stride: int,
    threshold: float = 0.6,
) -> list[int]:
    """Run sliding window inference and return deduplicated predicted labels."""
    T = len(keypoints)
    predictions: list[tuple[int, float]] = []

    for start in range(0, max(1, T - window + 1), stride):
        end = start + window
        chunk = keypoints[start:end]
        if len(chunk) < window:
            pad = np.zeros((window - len(chunk), 75, 3), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        inp_name = session.get_inputs()[0].name
        logits = session.run(None, {inp_name: chunk[np.newaxis].astype(np.float32)})[0]

        probs = _softmax(logits[0])
        best_cls = int(np.argmax(probs))
        best_prob = float(probs[best_cls])

        if best_prob >= threshold:
            predictions.append((best_cls, best_prob))

    # Deduplicate consecutive identical predictions
    result: list[int] = []
    prev = -1
    for cls, _ in predictions:
        if cls != prev:
            result.append(cls)
            prev = cls
    return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _effective_pps(window: int, stride: int, latency_ms: float) -> float:
    """Predict throughput in gesture/second given stride and latency per window."""
    # Each stride covers `stride` frames at 30 FPS → stride/30 seconds per call
    if latency_ms <= 0:
        return 0.0
    fps = 30
    time_covered_s = stride / fps
    time_per_call_s = latency_ms / 1000
    # We need one call per stride worth of video; calls can overlap
    return time_covered_s / max(time_covered_s, time_per_call_s)


# ── Single (window, stride) evaluation ───────────────────────────────────────

def _eval_config(
    session: Any,
    test_samples: list[Any],
    dataset: Any,
    window: int,
    stride: int,
    n_samples: int,
    threshold: float,
) -> dict[str, float]:
    correct = 0
    latencies: list[float] = []

    for sample in test_samples[:n_samples]:
        # Load a longer sequence (2× window) so sliding window has material to slide over
        full_len = max(window * 2, 60)
        kp = dataset.load_keypoints_mediapipe(sample, target_len=full_len)

        t0 = time.perf_counter()
        preds = sliding_window_predict(session, kp, window, stride, threshold)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Top-1: first non-null prediction matches label
        if preds and preds[0] == sample.label:
            correct += 1

    if not latencies:
        return {}

    latencies.sort()
    n = len(latencies)
    mean_lat = sum(latencies) / n
    p95 = latencies[int(0.95 * n)]

    return {
        "top1_accuracy":   correct / n_samples,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms":  p95,
        "effective_pps":   _effective_pps(window, stride, mean_lat),
        "window_size":     float(window),
        "stride":          float(stride),
        "overlap_ratio":   (window - stride) / window,
    }


# ── Synthetic dataset ─────────────────────────────────────────────────────────

class _SyntheticDataset:
    def __init__(self, n: int = 100, n_classes: int = 100):
        self.num_classes = n_classes
        self._n = n
        class _S:
            def __init__(self, label: int):
                self.label = label
                self.split = "test"
        self._samples = [_S(i % n_classes) for i in range(n)]

    def by_split(self, _: str) -> list[Any]:
        return self._samples

    def load_keypoints_mediapipe(self, _: Any, target_len: int = 60) -> np.ndarray:
        return np.random.randn(target_len, 75, 3).astype(np.float32)


class _RandomSession:
    """Fake ONNX session returning random logits."""
    def __init__(self, n_classes: int = 100):
        self._n = n_classes
        class _Input:
            name = "input"
        self._inputs = [_Input()]
    def get_inputs(self) -> list[Any]:
        return self._inputs
    def run(self, _: Any, __: Any) -> list[np.ndarray]:
        return [np.random.randn(1, self._n)]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        dataset = _SyntheticDataset(n_samples=args.n_samples)
        session = _RandomSession()
        print(f"[dry-run] Synthetic mode: {args.n_samples} samples")
    else:
        import onnxruntime as ort  # type: ignore[import]
        from experiments.shared.slovo_dataset import SlovoDataset

        dataset = SlovoDataset(args.slovo_root, split="all")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(args.onnx_model, sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        print(f"Slovo: {dataset.num_classes} classes")

    test_samples = dataset.by_split("test")
    all_results: dict[str, Any] = {}

    with mlflow_run(
        EXPERIMENT_NAME,
        run_name=f"window_search_n{args.n_samples}",
        tags={"experiment": "02"},
    ) as run:
        log_params({"n_samples": args.n_samples, "threshold": args.threshold,
                    "window_sizes": str(WINDOW_SIZES), "strides": str(STRIDES)})

        for window in WINDOW_SIZES:
            for stride in STRIDES:
                if stride > window:
                    continue  # stride must not exceed window
                key = f"w{window}_s{stride}"
                print(f"  window={window:2d}  stride={stride:2d} ... ", end="", flush=True)

                metrics = _eval_config(
                    session, test_samples, dataset,
                    window=window, stride=stride,
                    n_samples=args.n_samples,
                    threshold=args.threshold,
                )
                all_results[key] = metrics

                log_metrics({f"{key}/{k}": v for k, v in metrics.items()})
                print(f"acc={metrics.get('top1_accuracy', 0):.3f}  "
                      f"p95={metrics.get('p95_latency_ms', 0):.1f}ms  "
                      f"pps={metrics.get('effective_pps', 0):.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(all_results, RESULTS_DIR / "results.json")
    _print_grid(all_results)
    return all_results


def _print_grid(results: dict[str, Any]) -> None:
    print("\n" + "═" * 75)
    print("EXPERIMENT 02 — SLIDING WINDOW GRID")
    print("═" * 75)
    print(f"{'Config':<12} {'Acc':>6} {'P95ms':>8} {'PPS':>7} {'Overlap':>9}")
    print("─" * 75)
    for key in sorted(results):
        m = results[key]
        if not isinstance(m, dict):
            continue
        marker = " ← our choice" if key == "w30_s15" else ""
        print(f"{key:<12} {m.get('top1_accuracy', 0):>6.3f} "
              f"{m.get('p95_latency_ms', 0):>8.1f} "
              f"{m.get('effective_pps', 0):>7.2f} "
              f"{m.get('overlap_ratio', 0):>9.2f}{marker}")
    print("═" * 75)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slovo-root",  default="data/slovo")
    p.add_argument("--onnx-model",  default="models/gesture_classifier.onnx")
    p.add_argument("--n-samples",   type=int, default=100)
    p.add_argument("--threshold",   type=float, default=0.6)
    p.add_argument("--dry-run",     action="store_true")
    run_experiment(p.parse_args())


if __name__ == "__main__":
    main()
