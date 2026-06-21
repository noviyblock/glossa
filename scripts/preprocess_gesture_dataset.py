"""Preprocess the Slovo RSL gesture dataset into train/val/test numpy arrays.

Reads raw keypoints (from SlovoDataset via DWPose) and applies the
full preprocessing pipeline: imputation → length standardisation → normalisation.
Saves compressed .npz splits that the training pipeline consumes.

DVC stage: preprocess_gesture_dataset (see dvc.yaml)

Usage:
    # With Slovo dataset
    python scripts/preprocess_gesture_dataset.py --slovo-root /data/slovo

    # Dry-run (synthetic data, no dataset needed)
    python scripts/preprocess_gesture_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.shared.preprocessing import PreprocessConfig, StratifiedSplitter, preprocess_sample


def _load_yaml_params(path: str = "params.yaml") -> dict:
    try:
        import yaml  # type: ignore[import]
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _print_split_stats(name: str, X: np.ndarray, y: np.ndarray, classes: list[str]) -> None:
    from collections import Counter
    counts = Counter(y.tolist())
    rare = sum(1 for v in counts.values() if v < 3)
    print(f"  {name:<8} {len(X):>6} samples  "
          f"shape={X.shape[1:]}  "
          f"classes={len(counts)}  "
          f"rare(<3)={rare}")


def run(
    slovo_root: str | None,
    out_dir: Path,
    cfg: PreprocessConfig,
    n_samples: int | None = None,
    dry_run: bool = False,
    seed: int = 42,
    label_mapping_path: str | None = None,
) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("[dry-run] Generating synthetic dataset (200 samples, 50 classes)")
        n = n_samples or 200
        n_cls = 50
        rng = np.random.default_rng(seed)
        all_kp = rng.random((n, cfg.target_len, 75, 3)).astype(np.float32)
        all_labels = rng.integers(0, n_cls, n).tolist()
        class_names = [f"class_{i:03d}" for i in range(n_cls)]
    else:
        if not slovo_root:
            raise ValueError("--slovo-root is required unless --dry-run is set")

        from experiments.shared.slovo_dataset import SlovoDataset
        print(f"Loading SlovoDataset from {slovo_root} ...")
        ds = SlovoDataset(slovo_root, label_mapping_path=label_mapping_path)
        class_names = [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]
        samples = ds.samples
        if n_samples:
            samples = samples[:n_samples]

        print(f"  {len(samples)} samples, {len(class_names)} classes")
        print("  Extracting keypoints via DWPose ...")

        all_kp: list[np.ndarray] = []
        all_labels: list[int] = []
        rng = np.random.default_rng(seed)

        for i, s in enumerate(samples):
            if i % 500 == 0:
                print(f"    {i}/{len(samples)} ...")
            try:
                kp = ds.load_keypoints_dwpose(s, target_len=None)  # raw length
                all_kp.append(kp)
                all_labels.append(s.label)
            except Exception as e:
                if i < 5:
                    print(f"    Warning: skipped sample {i}: {e}")

        if not all_kp:
            raise RuntimeError("No keypoints extracted — check Slovo path and DWPose ONNX weights")

        print(f"  Extracted {len(all_kp)} keypoint sequences")
        all_kp = np.stack([
            preprocess_sample(kp, cfg, training=False)
            for kp in all_kp
        ])

    # Stratified split
    splitter = StratifiedSplitter(train=0.80, val=0.10, seed=seed)
    tr_idx, va_idx, te_idx = splitter.split(
        all_labels if isinstance(all_labels, list) else all_labels.tolist()
    )

    X_arr = all_kp if isinstance(all_kp, np.ndarray) else np.stack(all_kp)
    y_arr = np.array(all_labels, dtype=np.int64)

    splits = {
        "train": (tr_idx, True),   # training=True → augmentation ON
        "val":   (va_idx, False),
        "test":  (te_idx, False),
    }

    rng = np.random.default_rng(seed)
    stats: dict = {"splits": {}, "preprocessing": {}}

    for split_name, (idxs, do_augment) in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if not idxs:
            print(f"  Warning: {split_name} split is empty")
            continue

        X_split = np.stack([
            preprocess_sample(X_arr[i], cfg, training=do_augment, rng=rng)
            for i in idxs
        ])
        y_split = y_arr[idxs]

        np.save(split_dir / "features.npy", X_split)
        np.save(split_dir / "labels.npy",   y_split)

        _print_split_stats(split_name, X_split, y_split, class_names)
        stats["splits"][split_name] = {
            "n_samples": len(idxs),
            "n_classes": int(np.unique(y_split).shape[0]),
            "shape": list(X_split.shape),
        }

    # Save class names at the processed/ root (shared across splits)
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2))

    # Copy / generate label_mapping.json so downstream stages (exp 09) can load it
    if label_mapping_path and Path(label_mapping_path).exists():
        import shutil
        shutil.copy2(label_mapping_path, out_dir / "label_mapping.json")
    else:
        id_map = {name: i for i, name in enumerate(class_names)}
        (out_dir / "label_mapping.json").write_text(
            json.dumps({"label_to_id": id_map, "num_classes": len(id_map)},
                       ensure_ascii=False, indent=2))

    # Record preprocessing config for reproducibility
    stats["preprocessing"] = {
        "target_len":       cfg.target_len,
        "normalize":        cfg.normalize,
        "impute_missing":   cfg.impute_missing,
        "length_mode":      cfg.length_mode,
        "flip_prob":        cfg.flip_prob,
        "scale_range":      list(cfg.scale_range),
        "noise_std":        cfg.noise_std,
        "time_warp":        cfg.time_warp,
        "n_classes":        len(class_names),
        "elapsed_s":        round(time.perf_counter() - t0, 1),
    }
    (out_dir / "preprocessing_stats.json").write_text(
        json.dumps(stats, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s → {out_dir}")
    print(f"  Train : {stats['splits'].get('train', {}).get('n_samples', 0)}")
    print(f"  Val   : {stats['splits'].get('val',   {}).get('n_samples', 0)}")
    print(f"  Test  : {stats['splits'].get('test',  {}).get('n_samples', 0)}")


if __name__ == "__main__":
    params = _load_yaml_params()
    g_params = params.get("gesture", {})
    d_params = params.get("data", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slovo-root",     default=None)
    p.add_argument("--out-dir",        default="data/gestures/processed")
    p.add_argument("--target-len",     type=int,   default=g_params.get("sequence_length", 30))
    p.add_argument("--flip-prob",      type=float, default=g_params.get("augmentation", {}).get("flip_prob", 0.5))
    p.add_argument("--noise-std",      type=float, default=g_params.get("augmentation", {}).get("noise_std", 0.01))
    p.add_argument("--label-mapping",  default=d_params.get("label_mapping_path", ""),
                   help="Path to label_mapping.json from exp 00 EDA")
    p.add_argument("--n-samples",      type=int,   default=None, help="Limit dataset size (debugging)")
    p.add_argument("--seed",           type=int,   default=d_params.get("random_seed", 42))
    p.add_argument("--dry-run",        action="store_true")
    args = p.parse_args()

    cfg = PreprocessConfig(
        target_len=args.target_len,
        flip_prob=args.flip_prob,
        noise_std=args.noise_std,
    )
    run(
        slovo_root=args.slovo_root,
        out_dir=Path(args.out_dir),
        cfg=cfg,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
        seed=args.seed,
        label_mapping_path=args.label_mapping or None,
    )
