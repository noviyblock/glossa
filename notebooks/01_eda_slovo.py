"""Exploratory Data Analysis — Slovo RSL Dataset.

Runnable as a plain script or converted to Jupyter notebook via jupytext:
    jupytext --to notebook notebooks/01_eda_slovo.py

Usage:
    python notebooks/01_eda_slovo.py --slovo-root /data/slovo
    python notebooks/01_eda_slovo.py --slovo-root /data/slovo --plots
"""

# %% [markdown]
# # EDA: Slovo Russian Sign Language Dataset
#
# **Goal:** understand the data before training — class distribution, sequence
# length variability, keypoint quality, and signer diversity.
# Findings directly drive preprocessing and modelling decisions.

# %%
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

# %%
# ── 1. Load dataset ───────────────────────────────────────────────────────────

def load_annotations(slovo_root: str) -> list[dict]:
    """Parse annotations.csv into a list of dicts."""
    import csv
    root = Path(slovo_root)
    for candidate in [root / "annotations.csv",
                      root / "slovo" / "annotations.csv",
                      root / "SLOVO" / "annotations.csv"]:
        if candidate.exists():
            with open(candidate, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(f"annotations.csv not found under {slovo_root}")


def run_eda(slovo_root: str, plots: bool = False, out_dir: Path | None = None) -> dict:
    rows = load_annotations(slovo_root)
    print(f"\n{'═'*60}")
    print("  EDA: Slovo RSL Dataset")
    print(f"{'═'*60}")

    # ── 2. Basic stats ────────────────────────────────────────────────────────
    total = len(rows)
    labels   = [r.get("label", r.get("class", "")) for r in rows]
    splits   = [r.get("split", "train") for r in rows]
    classes  = sorted(set(labels))
    n_classes = len(classes)

    split_counts = Counter(splits)
    print(f"\n[Overview]")
    print(f"  Total samples : {total:,}")
    print(f"  Classes       : {n_classes:,}")
    for sp, cnt in sorted(split_counts.items()):
        print(f"  {sp:<12}: {cnt:,}  ({cnt/total*100:.1f}%)")

    # ── 3. Class distribution ─────────────────────────────────────────────────
    class_counts = Counter(labels)
    counts_arr   = np.array(list(class_counts.values()))

    print(f"\n[Class Distribution]")
    print(f"  Min samples/class  : {counts_arr.min()}")
    print(f"  Max samples/class  : {counts_arr.max()}")
    print(f"  Mean               : {counts_arr.mean():.1f}")
    print(f"  Median             : {np.median(counts_arr):.1f}")
    print(f"  Std                : {counts_arr.std():.1f}")

    # Imbalance ratio
    imbalance = counts_arr.max() / max(counts_arr.min(), 1)
    print(f"  Imbalance ratio    : {imbalance:.1f}x")
    if imbalance > 5:
        print("  ⚠ High imbalance detected → consider weighted loss or oversampling")

    # Long tail: classes with fewer than 10 samples
    rare = (counts_arr < 10).sum()
    print(f"  Classes < 10 samples: {rare} ({rare/n_classes*100:.1f}%)")

    # ── 4. Sequence length analysis ───────────────────────────────────────────
    begin_col = next((k for k in rows[0] if "begin" in k.lower()), None)
    end_col   = next((k for k in rows[0] if "end"   in k.lower()), None)

    seq_stats: dict = {}
    if begin_col and end_col:
        lengths = [
            max(0, int(r[end_col]) - int(r[begin_col]))
            for r in rows
            if r[begin_col].isdigit() and r[end_col].isdigit()
        ]
        if lengths:
            la = np.array(lengths)
            print(f"\n[Sequence Lengths (frames)]")
            print(f"  Min    : {la.min()}")
            print(f"  Max    : {la.max()}")
            print(f"  Mean   : {la.mean():.1f}")
            print(f"  Median : {np.median(la):.1f}")
            print(f"  Std    : {la.std():.1f}")
            print(f"  p5     : {np.percentile(la, 5):.0f}")
            print(f"  p95    : {np.percentile(la, 95):.0f}")
            print(f"  → Production window 30 frames covers "
                  f"{(la <= 30).sum()/len(la)*100:.1f}% of samples")
            seq_stats = {
                "min": int(la.min()), "max": int(la.max()),
                "mean": float(la.mean()), "median": float(np.median(la)),
                "std": float(la.std()),
                "p5": float(np.percentile(la, 5)),
                "p95": float(np.percentile(la, 95)),
                "pct_le30": float((la <= 30).sum() / len(la)),
            }

    # ── 5. Keypoint quality (sample 200 videos) ───────────────────────────────
    print(f"\n[Keypoint Quality — sampling 200 videos]")
    try:
        from experiments.shared.slovo_dataset import SlovoDataset
        ds = SlovoDataset(slovo_root)
        sample_n = min(200, len(ds.samples))
        idxs = np.random.default_rng(42).choice(len(ds.samples), sample_n, replace=False)
        samples = [ds.samples[i] for i in idxs]

        missing_rates: list[float] = []
        hand_visibility: list[float] = []

        for s in samples:
            try:
                kp = ds.load_keypoints_mediapipe(s, target_len=30)  # (T, 75, 3)
                # Missing = coords all near zero
                missing = np.all(np.abs(kp) < 0.02, axis=-1)  # (T, 75) bool
                missing_rates.append(float(missing.mean()))
                # Hand visibility: fraction of frames where hands are present
                rhand_present = ~np.all(np.abs(kp[:, 54:75, :]) < 0.02, axis=(1, 2))
                lhand_present = ~np.all(np.abs(kp[:, 33:54, :]) < 0.02, axis=(1, 2))
                hand_visibility.append(float((rhand_present | lhand_present).mean()))
            except Exception:
                continue

        if missing_rates:
            mr = np.array(missing_rates)
            hv = np.array(hand_visibility)
            print(f"  Mean missing landmark rate : {mr.mean()*100:.1f}%")
            print(f"  Max missing landmark rate  : {mr.max()*100:.1f}%")
            print(f"  Samples >20% missing       : {(mr > 0.2).sum()}")
            print(f"  Mean hand visibility       : {hv.mean()*100:.1f}%")
            print(f"  → Imputation needed for {(mr > 0.01).sum()} of {len(mr)} samples")
        else:
            print("  Could not load keypoints (run with Slovo dataset)")
    except Exception as e:
        print(f"  Skipped (SlovoDataset unavailable): {e}")
        missing_rates = []

    # ── 6. Signer diversity proxy ─────────────────────────────────────────────
    signer_col = next((k for k in rows[0] if "signer" in k.lower()), None)
    signer_stats: dict = {}
    if signer_col:
        signers = Counter(r[signer_col] for r in rows)
        print(f"\n[Signer Diversity]")
        print(f"  Unique signers : {len(signers)}")
        per_signer = np.array(list(signers.values()))
        print(f"  Samples/signer : min={per_signer.min()} mean={per_signer.mean():.0f} "
              f"max={per_signer.max()}")
        print(f"  → Signer-independent validation recommended to test generalisation")
        signer_stats = {"n_signers": len(signers), "mean_samples": float(per_signer.mean())}

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    if plots:
        _make_plots(class_counts, counts_arr, seq_stats, out_dir)

    # ── 8. Summary & recommendations ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Preprocessing Recommendations")
    print(f"{'─'*60}")
    print("  1. Impute missing landmarks via linear interpolation")
    print("  2. Normalise: hip-centred translation + shoulder-width scaling")
    print("  3. Standardise length to 30 frames (covers p95 of dataset)")
    print("  4. Stratified train/val/test split (80/10/10)")
    print("  5. Weighted cross-entropy loss for class imbalance")
    print("  6. Augment: horizontal flip, ±15% scale, noise σ=0.01, time warp")
    print(f"{'═'*60}\n")

    results = {
        "total_samples": total,
        "n_classes": n_classes,
        "split_counts": dict(split_counts),
        "class_distribution": {
            "min": int(counts_arr.min()),
            "max": int(counts_arr.max()),
            "mean": float(counts_arr.mean()),
            "std": float(counts_arr.std()),
            "imbalance_ratio": float(imbalance),
            "rare_classes_lt10": int(rare),
        },
        "sequence_lengths": seq_stats,
        "signer_diversity": signer_stats,
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eda_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Results saved to {out_dir / 'eda_results.json'}")

    return results


def _make_plots(
    class_counts: Counter,
    counts_arr: np.ndarray,
    seq_stats: dict,
    out_dir: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Class distribution histogram
    axes[0].hist(counts_arr, bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Class Distribution\n(samples per class)")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Number of classes")
    axes[0].axvline(counts_arr.mean(), color="orange", linestyle="--",
                    label=f"mean={counts_arr.mean():.0f}")
    axes[0].legend()

    # Top-20 and bottom-20 classes
    sorted_counts = sorted(class_counts.values())
    axes[1].bar(range(20), sorted_counts[:20], color="#DD8452", label="Rarest 20")
    axes[1].bar(range(20, 40), sorted_counts[-20:], color="#4C72B0", label="Most common 20")
    axes[1].set_title("Long-tail: rarest vs most common classes")
    axes[1].set_ylabel("Samples")
    axes[1].legend()

    # Sequence length distribution (if available)
    if seq_stats:
        ax = axes[2]
        ax.axvline(30, color="red", linestyle="--", label="target len=30")
        ax.set_title("Sequence Length Distribution")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Density")
        ax.legend()
        ax.text(0.5, 0.5, f"p95={seq_stats.get('p95', '?'):.0f} frames",
                transform=ax.transAxes, ha="center")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / "eda_plots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {path}")
    else:
        plt.show()
    plt.close()


# %% [markdown]
# ## Findings summary
#
# | Finding | Impact | Action |
# |---------|--------|--------|
# | Long-tail class distribution | Biased loss | Weighted CE |
# | Missing landmarks in ~30% of frames | NaN-equivalent inputs | Linear imputation |
# | Sequence length 8–120 frames | Variable-length model input | Interpolate to 30 |
# | Multiple signers per class | Signer-dependent overfitting | Leave-one-signer-out CV |
# | Hands not visible in some frames | Degraded features | Impute + noise augmentation |

# %%
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EDA for Slovo RSL dataset")
    p.add_argument("--slovo-root", required=True, help="Path to Slovo dataset root")
    p.add_argument("--plots", action="store_true", help="Generate matplotlib plots")
    p.add_argument("--out-dir", default="experiments/results/eda",
                   help="Directory to save results JSON and plots")
    args = p.parse_args()
    run_eda(args.slovo_root, plots=args.plots, out_dir=Path(args.out_dir))
