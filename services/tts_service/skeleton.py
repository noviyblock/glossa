from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from config import PROCESSED_GESTURES_DIR

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class SkeletonSequenceProvider:
    """gloss_sequence -> per-gloss keypoint sequences for client-side
    skeleton playback (see clients/web's _skeletonPanel) -- an alternative
    to SignVideoAssembler's concatenated-clip video, reusing the same
    per-gesture keypoint data the ST-GCN classifier trains on
    (data/gestures/processed_64_200) instead of separately-recorded video
    clips: much smaller payload, no ffmpeg concat step, and (unlike the
    video path) the underlying data has full vocabulary coverage since
    it's the same set the classifier was evaluated against.

    Matching mirrors SignVideoAssembler's normalized/greedy-longest-phrase
    approach for consistency -- duplicated (not shared), it's ~10 lines and
    the two providers have different underlying data sources.
    """

    def __init__(self, processed_dir: str = PROCESSED_GESTURES_DIR, split: str = "val") -> None:
        base = Path(processed_dir) / split
        self._features = np.load(base / "features.npy", mmap_mode="r")
        labels = np.load(base / "labels.npy")
        class_names: list[str] = json.loads(
            (Path(processed_dir) / "class_names.json").read_text(encoding="utf-8")
        )

        self._norm_to_sample_idx: dict[str, int] = {}
        for i, label in enumerate(labels):
            name = class_names[int(label)]
            key = self._normalize(name)
            self._norm_to_sample_idx.setdefault(key, i)  # first sample wins, deterministic
        self._max_words = max((len(k.split()) for k in self._norm_to_sample_idx), default=1)
        logger.info(
            "SkeletonSequenceProvider: %d/%d classes have a %s-split sample (processed_dir=%s)",
            len(self._norm_to_sample_idx), len(class_names), split, processed_dir,
        )

    @staticmethod
    def _normalize(s: str) -> str:
        return _PUNCT_RE.sub("", s).strip().upper()

    def get(self, gloss_sequence: str) -> list[dict]:
        """[{"gloss": str, "frames": [[[x,y,conf], ...75...], ...T...]}, ...]
        for each (space-separated) gloss in the sequence that matched a
        sample -- best-effort like SignVideoAssembler.build: unmatched
        tokens are skipped, not a hard failure."""
        tokens = gloss_sequence.strip().split()
        results: list[dict] = []
        i = 0
        while i < len(tokens):
            matched = False
            max_span = min(self._max_words, len(tokens) - i)
            for span in range(max_span, 0, -1):
                phrase = self._normalize(" ".join(tokens[i:i + span]))
                idx = self._norm_to_sample_idx.get(phrase)
                if idx is None:
                    continue
                results.append({
                    "gloss": " ".join(tokens[i:i + span]),
                    "frames": self._to_screen_space(self._features[idx]).tolist(),
                })
                i += span
                matched = True
                break
            if not matched:
                i += 1
        return results

    @staticmethod
    def _to_screen_space(clip: np.ndarray) -> np.ndarray:
        """(T, 75, 3) hip/shoulder-normalized -> (T, 75, 3) with x/y remapped
        into [0,1] (matching what the client's _SkeletonPainter already
        expects from the live-camera path). Fit ONCE over the whole clip's
        bounding box, not per-frame, so the figure doesn't rescale/"breathe"
        frame to frame. Confidence (index 2) passes through unchanged; a
        point that was (0,0) — the existing zero-confidence/padding
        convention — stays exactly (0,0,*) so the client's existing
        skip-if-zero rendering keeps working unmodified.
        """
        out = clip.astype(np.float32).copy()
        xy = out[:, :, :2]
        conf = out[:, :, 2]
        valid = (conf > 0) & ~((xy[:, :, 0] == 0) & (xy[:, :, 1] == 0))
        if not valid.any():
            return out
        xs = xy[:, :, 0][valid]
        ys = xy[:, :, 1][valid]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        margin = 0.1
        out[:, :, 0] = np.where(valid, (xy[:, :, 0] - x_min) / x_span * (1 - 2 * margin) + margin, 0.0)
        out[:, :, 1] = np.where(valid, (xy[:, :, 1] - y_min) / y_span * (1 - 2 * margin) + margin, 0.0)
        return out
