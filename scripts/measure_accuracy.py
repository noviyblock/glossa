"""Offline accuracy measurement: run real recorded clips through the ACTUAL
production cv_service pipeline (imported directly, not reimplemented) and
report top-1/top-3 accuracy against known labels — replaces "on the glass"
guessing about whether a prediction is close or wildly off.

Two input modes:

1. Raw video clips + label CSV (real camera recordings — the main use case):
    python scripts/measure_accuracy.py \
        --clips-dir path/to/clips --labels-csv path/to/labels.csv

    labels.csv columns: filename,gloss
    (gloss must match a value in data/idx_to_class.json)

2. Pre-extracted keypoint sequences (.npy, raw (T,75,3) per file, not yet
   resampled/normalized — e.g. output from colab_glossa_04's extraction):
    python scripts/measure_accuracy.py \
        --keypoints-dir path/to/npy --labels-csv path/to/labels.csv

    labels.csv columns: filename,gloss  (filename without .npy extension ok)

Reports: top-1/top-3 accuracy, with/without TTA, per-clip breakdown, avg
extraction latency/frame (rtmlib forward pass under whatever RTMLIB_MODE
the process was started with -- set the env var before running to compare
modes, e.g. RTMLIB_MODE=balanced python scripts/measure_accuracy.py ...),
and flags whether wrong predictions are at least in the SAME top-3
(near-miss -- suggests real model confusion between similar signs) vs
nowhere close (suggests a pipeline bug rather than a hard example).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_CV_SERVICE_DIR = Path(__file__).resolve().parent.parent / "services" / "cv_service"
sys.path.insert(0, str(_CV_SERVICE_DIR))


def load_labels_csv(path: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["filename"]).stem
            labels[stem] = row["gloss"].strip()
    return labels


def extract_raw_keypoints_from_video(extractor, video_path: str) -> tuple[np.ndarray, float]:
    """Run the ACTUAL production KeypointExtractor over every frame of a clip.

    Returns (keypoints, extraction_seconds) -- the timer wraps only
    extractor.extract() calls (the rtmlib Wholebody forward pass under
    whatever RTMLIB_MODE the process was started with), not cv2 decode,
    so it's comparable to the cv_service /process_frame latency budget.
    """
    import cv2
    from keypoint_extractor import TrackState

    cap = cv2.VideoCapture(video_path)
    track = TrackState()
    frames_kp = []
    elapsed = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        kp = extractor.extract(frame, track)
        elapsed += time.perf_counter() - t0
        frames_kp.append(kp)
    cap.release()
    if not frames_kp:
        return np.zeros((0, 75, 3), dtype=np.float32), 0.0
    return np.stack(frames_kp, axis=0), elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", help="directory of .mp4/.mov/.avi clips (mode 1)")
    ap.add_argument("--keypoints-dir", help="directory of raw (T,75,3) .npy files (mode 2)")
    ap.add_argument("--labels-csv", required=True, help="filename,gloss CSV")
    ap.add_argument("--norm-stats", default="models/norm_stats.npz")
    ap.add_argument("--ov-xml",   default="models/stgcn_topk_int8/stgcn_topk_int8.xml")
    ap.add_argument("--onnx",     default="models/gesture_classifier_mobile.onnx")
    ap.add_argument("--class-map", default="data/idx_to_class.json")
    ap.add_argument("--min-frames", type=int, default=8, help="skip clips shorter than this after extraction")
    args = ap.parse_args()

    if not args.clips_dir and not args.keypoints_dir:
        ap.error("pass either --clips-dir or --keypoints-dir")

    import config as cfg
    from gesture_classifier import GestureClassifier
    from normalizer import Normalizer
    from sliding_window import _resample_to

    labels = load_labels_csv(args.labels_csv)
    print(f"Loaded {len(labels)} labels from {args.labels_csv}")

    normalizer = Normalizer(args.norm_stats)
    classifier = GestureClassifier(
        ov_xml_path=args.ov_xml, onnx_path=args.onnx, class_map_path=args.class_map,
    )
    print(f"Classifier backend: {classifier._backend}")  # diagnostic only

    extractor = None
    if args.clips_dir:
        from keypoint_extractor import KeypointExtractor
        extractor = KeypointExtractor()

    class_to_idx = {v: k for k, v in classifier._idx_to_class.items()}

    rows: list[tuple[str, str]] = []  # (stem, true_gloss)
    if args.clips_dir:
        clips_dir = Path(args.clips_dir)
        for ext in ("*.mp4", "*.mov", "*.avi"):
            for p in clips_dir.glob(ext):
                if p.stem in labels:
                    rows.append((p.stem, labels[p.stem]))
    else:
        kp_dir = Path(args.keypoints_dir)
        for p in kp_dir.glob("*.npy"):
            if p.stem in labels:
                rows.append((p.stem, labels[p.stem]))

    if not rows:
        print("No clips/keypoint files matched entries in the labels CSV — nothing to evaluate.")
        return

    unknown_gloss = sorted({g for _, g in rows if g not in class_to_idx})
    if unknown_gloss:
        print(f"WARNING: {len(unknown_gloss)} labels don't match any class in {args.class_map}, skipping those rows:")
        for g in unknown_gloss[:10]:
            print(f"  {g!r}")
        rows = [(s, g) for s, g in rows if g in class_to_idx]

    print(f"Evaluating {len(rows)} labeled clips...\n")

    top1_hits = top1_hits_tta = top3_hits = near_miss = 0
    skipped = 0
    total_latency = 0.0
    total_extract_latency = 0.0
    total_extract_frames = 0
    mistakes: list[tuple[str, str, str, bool]] = []  # (stem, true, pred, near_miss)

    for stem, true_gloss in rows:
        if args.clips_dir:
            video_path = str(next(Path(args.clips_dir).glob(f"{stem}.*")))
            raw_kp, extract_elapsed = extract_raw_keypoints_from_video(extractor, video_path)
            total_extract_latency += extract_elapsed
            total_extract_frames += raw_kp.shape[0]
        else:
            raw_kp = np.load(Path(args.keypoints_dir) / f"{stem}.npy").astype(np.float32)

        if raw_kp.shape[0] < args.min_frames:
            skipped += 1
            continue

        window = _resample_to(raw_kp, cfg.WINDOW_SIZE)
        norm_win = normalizer(window)

        t0 = time.perf_counter()
        results = classifier.predict_top3(norm_win)
        total_latency += time.perf_counter() - t0
        results_tta = classifier.predict_top3_tta(norm_win) if cfg.TTA_ENABLED else results

        top1 = results[0]["gloss"]
        top1_tta = results_tta[0]["gloss"]
        top3_glosses = [r["gloss"] for r in results_tta]

        hit1 = top1 == true_gloss
        hit1_tta = top1_tta == true_gloss
        hit3 = true_gloss in top3_glosses

        top1_hits += hit1
        top1_hits_tta += hit1_tta
        top3_hits += hit3
        if not hit1_tta:
            is_near = hit3  # wrong top-1 but true label still in top-3 -> plausible confusion, not garbage
            near_miss += is_near
            mistakes.append((stem, true_gloss, top1_tta, is_near))

    n = len(rows) - skipped
    if n == 0:
        print("Every clip was skipped (too short after extraction) — nothing to report.")
        return

    print(f"=== Results ({n} clips evaluated, {skipped} skipped as too short) ===")
    print(f"Top-1 accuracy (single pass): {top1_hits/n:.4f}")
    print(f"Top-1 accuracy (with TTA):    {top1_hits_tta/n:.4f}")
    print(f"Top-3 accuracy (with TTA):    {top3_hits/n:.4f}")
    print(f"Avg single-pass latency:      {1000*total_latency/n:.2f}ms")
    if total_extract_frames:
        print(f"Avg extraction latency/frame: {1000*total_extract_latency/total_extract_frames:.2f}ms "
              f"(rtmlib mode={cfg.RTMLIB_MODE}, device={cfg.RTMLIB_DEVICE}, "
              f"{total_extract_frames} frames over {n} clips)")
    wrong = n - top1_hits_tta
    if wrong:
        print(f"\nOf {wrong} wrong top-1 predictions: {near_miss} ({near_miss/wrong*100:.0f}%) had the "
              f"true gloss in top-3 (near-miss — model genuinely confused two similar signs), "
              f"{wrong-near_miss} ({(wrong-near_miss)/wrong*100:.0f}%) missed entirely "
              f"(true gloss not even in top-3 — investigate the pipeline, not just the model).")

    if mistakes:
        print("\nMistakes (clip / true / predicted / near-miss?):")
        for stem, true_g, pred_g, is_near in mistakes[:30]:
            flag = "near-miss" if is_near else "FAR OFF"
            print(f"  {stem:20s}  true={true_g!r:20s}  pred={pred_g!r:20s}  [{flag}]")


if __name__ == "__main__":
    main()
