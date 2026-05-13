"""Experiment 09 — Model Validation: Stratified K-Fold, Learning Curves, Calibration.

Answers three validation questions required for a rigorous DS pipeline:

  1. Cross-validation   — is the accuracy estimate stable across folds?
                          Reports mean ± 95% CI for accuracy, F1, top-5.

  2. Learning curve     — how does performance scale with training set size?
                          Detects overfitting and whether more data would help.

  3. Confidence calibration — does the model's softmax confidence match
                          its actual accuracy? Computes Expected Calibration
                          Error (ECE) and reliability diagram data.

Model: lightweight MLP on flattened keypoints (can swap for STGCN ONNX).
Dataset: processed Slovo splits in data/gestures/processed/ or --dry-run.

Usage:
    python -m experiments.09_cross_validation.run --dry-run
    python -m experiments.09_cross_validation.run \
        --data-dir data/gestures/processed \
        --k-folds 5
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

from experiments.shared.mlflow_utils import log_metrics, log_params, mlflow_run, save_results

EXPERIMENT_NAME = "09_cross_validation"
RESULTS_DIR = Path("experiments/results/09_cross_validation")


# ── Lightweight classifier (swap for STGCN in real run) ──────────────────────

def _build_mlp(input_dim: int, n_classes: int) -> Any:
    try:
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, n_classes),
                )
            def forward(self, x: Any) -> Any:  # type: ignore[override]
                return self.net(x)

        return MLP()
    except ImportError as e:
        raise ImportError("pip install torch") from e


def _train_eval_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> dict[str, float]:
    """Train one fold and return validation metrics."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    input_dim = X_train.shape[1]
    model = _build_mlp(input_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        sch.step()

    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32)
        logits = model(Xv)
        probs  = torch.softmax(logits, dim=-1).numpy()
        preds  = logits.argmax(dim=-1).numpy()

    acc = float((preds == y_val).mean())
    top5 = float(np.mean([
        y_val[i] in np.argsort(probs[i])[-5:]
        for i in range(len(y_val))
    ]))

    # Per-class F1 (macro)
    from collections import defaultdict
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    for true, pred in zip(y_val.tolist(), preds.tolist()):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    f1s = []
    for c in range(n_classes):
        p = tp[c] / (tp[c] + fp[c] + 1e-9)
        r = tp[c] / (tp[c] + fn[c] + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    f1_macro = float(np.mean(f1s))

    # ECE (Expected Calibration Error, 10 bins)
    ece = _compute_ece(probs, y_val, n_bins=10)

    return {
        "accuracy": acc,
        "top5_accuracy": top5,
        "f1_macro": f1_macro,
        "ece": ece,
        "n_val": float(len(y_val)),
    }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: measures alignment of confidence and accuracy."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(avg_conf - avg_acc)
    return float(ece)


def _ci_95(values: list[float]) -> tuple[float, float]:
    """95% CI assuming t-distribution (for small k)."""
    import math
    n = len(values)
    if n < 2:
        return (values[0], values[0])
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    # t critical for n-1 df, 95% two-tailed (approximation)
    t_crit = {1: 12.7, 2: 4.3, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
              7: 2.36, 8: 2.31, 9: 2.26}.get(n - 1, 2.0)
    margin = t_crit * std / math.sqrt(n)
    return (round(mean - margin, 4), round(mean + margin, 4))


# ── Learning curve ────────────────────────────────────────────────────────────

def _learning_curve(
    X: np.ndarray, y: np.ndarray,
    n_classes: int,
    train_fracs: list[float],
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train on increasing fractions of data, report val accuracy."""
    rng = np.random.default_rng(seed)
    n = len(X)
    val_size = max(50, int(n * 0.15))
    perm = rng.permutation(n)
    val_idx   = perm[:val_size]
    pool_idx  = perm[val_size:]

    X_val, y_val = X[val_idx], y[val_idx]
    curve = []
    for frac in train_fracs:
        n_tr = max(n_classes, int(len(pool_idx) * frac))
        tr_idx = pool_idx[:n_tr]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        try:
            m = _train_eval_fold(X_tr, y_tr, X_val, y_val, n_classes, epochs=20)
            curve.append({"train_size": n_tr, "val_accuracy": m["accuracy"],
                          "f1_macro": m["f1_macro"]})
        except Exception as e:
            curve.append({"train_size": n_tr, "error": str(e)})
    return curve


# ── Simulated results ─────────────────────────────────────────────────────────

_SIM_FOLDS = [
    {"accuracy": 0.871, "top5_accuracy": 0.963, "f1_macro": 0.854, "ece": 0.042},
    {"accuracy": 0.868, "top5_accuracy": 0.961, "f1_macro": 0.851, "ece": 0.045},
    {"accuracy": 0.876, "top5_accuracy": 0.965, "f1_macro": 0.860, "ece": 0.039},
    {"accuracy": 0.863, "top5_accuracy": 0.958, "f1_macro": 0.846, "ece": 0.048},
    {"accuracy": 0.872, "top5_accuracy": 0.962, "f1_macro": 0.855, "ece": 0.041},
]

_SIM_CURVE = [
    {"train_size": 200,  "val_accuracy": 0.61, "f1_macro": 0.58},
    {"train_size": 500,  "val_accuracy": 0.72, "f1_macro": 0.69},
    {"train_size": 1000, "val_accuracy": 0.80, "f1_macro": 0.77},
    {"train_size": 2000, "val_accuracy": 0.85, "f1_macro": 0.83},
    {"train_size": 4000, "val_accuracy": 0.87, "f1_macro": 0.85},
    {"train_size": 8000, "val_accuracy": 0.88, "f1_macro": 0.87},
]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}

    if args.dry_run:
        fold_metrics = _SIM_FOLDS[:args.k_folds]
        curve = _SIM_CURVE
        print(f"[dry-run] Simulated {args.k_folds}-fold CV:")
        for i, m in enumerate(fold_metrics):
            print(f"  Fold {i+1}: acc={m['accuracy']:.3f}  "
                  f"top5={m['top5_accuracy']:.3f}  "
                  f"f1={m['f1_macro']:.3f}  ECE={m['ece']:.3f}")
    else:
        data_dir = Path(args.data_dir)
        try:
            X_all = np.concatenate([
                np.load(data_dir / s / "features.npy")
                for s in ["train", "val", "test"]
                if (data_dir / s / "features.npy").exists()
            ])
            y_all = np.concatenate([
                np.load(data_dir / s / "labels.npy")
                for s in ["train", "val", "test"]
                if (data_dir / s / "labels.npy").exists()
            ])
        except FileNotFoundError:
            print("Processed data not found — use --dry-run or run preprocess_gesture_dataset.py")
            return {}

        n_classes = int(y_all.max()) + 1
        X_flat = X_all.reshape(len(X_all), -1).astype(np.float32)

        print(f"Loaded {len(X_flat)} samples, {n_classes} classes, "
              f"feature dim={X_flat.shape[1]}")

        # Stratified K-Fold
        fold_metrics = []
        rng = np.random.default_rng(args.seed)
        from collections import defaultdict
        class_buckets: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(y_all.tolist()):
            class_buckets[lbl].append(i)

        folds: list[list[int]] = [[] for _ in range(args.k_folds)]
        for idxs in class_buckets.values():
            perm = list(rng.permutation(idxs))
            for i, idx in enumerate(perm):
                folds[i % args.k_folds].append(idx)

        for fold_i, val_idx in enumerate(folds):
            tr_idx = [i for j, f in enumerate(folds) if j != fold_i for i in f]
            print(f"  Fold {fold_i+1}/{args.k_folds} "
                  f"(train={len(tr_idx)}, val={len(val_idx)}) ...", end=" ", flush=True)
            t0 = time.perf_counter()
            m = _train_eval_fold(
                X_flat[tr_idx], y_all[tr_idx],
                X_flat[val_idx], y_all[val_idx],
                n_classes, epochs=args.epochs,
            )
            print(f"acc={m['accuracy']:.3f}  ECE={m['ece']:.3f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            fold_metrics.append(m)

        # Learning curve
        print("\nLearning curve ...")
        fracs = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
        curve = _learning_curve(X_flat, y_all, n_classes, fracs, seed=args.seed)

    # Aggregate fold stats
    for metric in ["accuracy", "top5_accuracy", "f1_macro", "ece"]:
        vals = [m[metric] for m in fold_metrics if metric in m]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        ci   = _ci_95(vals)
        print(f"  {metric:<18} {mean:.4f} ± {std:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        results[f"{metric}_mean"] = mean
        results[f"{metric}_std"]  = std
        results[f"{metric}_ci_lo"] = ci[0]
        results[f"{metric}_ci_hi"] = ci[1]

    results["fold_details"] = fold_metrics
    results["learning_curve"] = curve

    # Print learning curve
    print(f"\n  {'Train size':>12}  {'Val Acc':>8}  {'F1 Macro':>10}")
    print("  " + "─" * 36)
    for pt in curve:
        if "error" in pt:
            print(f"  {pt['train_size']:>12}  ERROR: {pt['error']}")
        else:
            print(f"  {pt['train_size']:>12}  {pt['val_accuracy']:>8.3f}  {pt['f1_macro']:>10.3f}")

    # Saturation check: if accuracy gain < 1% from 75% to 100% train size
    if len(curve) >= 2:
        last_two = [pt["val_accuracy"] for pt in curve[-2:] if "val_accuracy" in pt]
        if len(last_two) == 2 and abs(last_two[1] - last_two[0]) < 0.01:
            print("\n  ✓ Learning curve saturated — more data unlikely to help significantly")
        else:
            print("\n  ↑ Learning curve still rising — more data would likely improve accuracy")

    with mlflow_run(
        EXPERIMENT_NAME,
        run_name=f"cv_k{args.k_folds}_seed{args.seed}",
        tags={"experiment": "09"},
    ) as _run:
        log_params({"k_folds": args.k_folds, "epochs": args.epochs,
                    "dry_run": str(args.dry_run)})
        log_metrics({k: v for k, v in results.items()
                     if isinstance(v, float) and "mean" in k or "ci" in k})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_copy = {k: v for k, v in results.items() if k != "fold_details"}
    save_copy["fold_details"] = fold_metrics
    save_results(save_copy, RESULTS_DIR / "results.json")
    _print_summary(results)
    return results


def _print_summary(results: dict[str, Any]) -> None:
    print("\n" + "═" * 65)
    print("EXPERIMENT 09 — CROSS-VALIDATION & VALIDATION REPORT")
    print("═" * 65)
    print(f"  {'Metric':<20} {'Mean':>8} {'±Std':>8} {'95% CI':>20}")
    print("  " + "─" * 58)
    for m in ["accuracy", "top5_accuracy", "f1_macro", "ece"]:
        mean = results.get(f"{m}_mean", 0)
        std  = results.get(f"{m}_std", 0)
        lo   = results.get(f"{m}_ci_lo", 0)
        hi   = results.get(f"{m}_ci_hi", 0)
        print(f"  {m:<20} {mean:>8.4f} {std:>8.4f}  [{lo:.4f}, {hi:.4f}]")
    print("═" * 65)
    print("  ECE < 0.05 → model is well-calibrated")
    print("  ECE > 0.10 → consider temperature scaling")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/gestures/processed")
    p.add_argument("--k-folds",  type=int, default=5)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--dry-run",  action="store_true")
    run_experiment(p.parse_args())


if __name__ == "__main__":
    main()
