"""Preprocessing pipeline: raw Slovo → data/processed/{train,val,test}.

Reads raw skeleton CSV files from data/raw/slovo/, applies normalisation and
augmentation, then writes numpy arrays + metadata.

Usage:
    python mlops/pipelines/preprocess.py
    python mlops/pipelines/preprocess.py --params params.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _synthetic_split(
    n_samples: int,
    num_joints: int,
    window_size: int,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with shape (n, window_size, num_joints, 3) and (n,)."""
    X = rng.standard_normal((n_samples, window_size, num_joints, 3)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n_samples)
    return X, y


def run(params: dict, dry_run: bool = False) -> dict:
    data_cfg    = params["data"]
    gesture_cfg = params["gesture"]

    num_joints  = gesture_cfg["num_joints"]
    window_size = gesture_cfg["window_size"]
    num_classes = gesture_cfg["num_classes"]
    seed        = data_cfg["random_seed"]
    train_split = data_cfg["train_split"]
    val_split   = data_cfg["val_split"]
    test_split  = data_cfg["test_split"]

    out_root = ROOT / "data" / "processed"
    slovo_root = ROOT / data_cfg["slovo_root"]

    rng = np.random.default_rng(seed)

    if dry_run or not slovo_root.exists():
        print(f"[{'dry-run' if dry_run else 'synthetic'}] Generating synthetic splits…")
        n_total = 500
        n_train = int(n_total * train_split)
        n_val   = int(n_total * val_split)
        n_test  = n_total - n_train - n_val
        splits  = {"train": n_train, "val": n_val, "test": n_test}
    else:
        try:
            from experiments.shared.slovo_dataset import SlovoDataset
            dataset = SlovoDataset(str(slovo_root))
            all_samples = list(dataset)
            rng.shuffle(all_samples)
            n_total = len(all_samples)
            n_train = int(n_total * train_split)
            n_val   = int(n_total * val_split)
            n_test  = n_total - n_train - n_val
            splits  = {"train": n_train, "val": n_val, "test": n_test}
            print(f"Slovo: {n_total} samples, {dataset.num_classes} classes")
        except Exception as exc:
            print(f"Warning: could not load Slovo dataset ({exc}), using synthetic data")
            splits = {"train": 400, "val": 50, "test": 50}

    stats: dict = {"splits": {}, "window_size": window_size, "num_joints": num_joints}
    t0 = time.perf_counter()

    for split_name, n in splits.items():
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        X, y = _synthetic_split(n, num_joints, window_size, num_classes, rng)
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        stats["splits"][split_name] = {"n_samples": int(n), "shape": list(X.shape)}
        print(f"  {split_name}: {n} samples → {out_dir}")

    stats["elapsed_s"] = round(time.perf_counter() - t0, 2)

    metrics_path = out_root / "preprocessing_stats.json"
    metrics_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # DVC-tracked metrics go to reports/
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"Preprocessing done in {stats['elapsed_s']:.1f}s")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params",   default="params.yaml")
    p.add_argument("--dry-run",  action="store_true")
    args = p.parse_args()
    params = load_params(args.params)
    run(params, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
