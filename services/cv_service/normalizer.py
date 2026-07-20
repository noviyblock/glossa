from __future__ import annotations

import numpy as np

from config import NORM_STATS_PATH

# Indices into the 75-point array (training body order, see keypoint_extractor.py)
# used by the offline preprocessing script (colab_glossa_00) to make coordinates
# translation- and scale-invariant before z-score normalisation.
_HIP_IDX  = [11, 12]
_SHOU_IDX = [5, 6]
_CLIP = 10.0


def _translate_scale_invariant(window: np.ndarray) -> np.ndarray:
    """Hip-center + shoulder-distance scale, mirrors colab_glossa_00's normalize_keypoints().

    window: (T, 75, 3) → (T, 75, 3), x/y channels made translation/scale invariant.
    """
    xy = window[:, :, :2].copy()

    hips = xy[:, _HIP_IDX, :]
    hips_center = np.median(hips, axis=1, keepdims=True)
    xy -= hips_center

    left_shoulder  = xy[:, _SHOU_IDX[0], :]
    right_shoulder = xy[:, _SHOU_IDX[1], :]
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6).reshape(-1, 1, 1)
    xy /= shoulder_dist

    out = window.copy()
    out[:, :, :2] = xy
    return out


class Normalizer:
    def __init__(self, stats_path: str = NORM_STATS_PATH) -> None:
        data = np.load(stats_path)
        self._mean = data["mean"].astype(np.float32)  # (1, 1, 1, 3), global per-channel
        self._std  = data["std"].astype(np.float32)   # (1, 1, 1, 3)
        self._std  = np.where(self._std < 1e-6, 1.0, self._std)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Normalise window of shape (T, 75, 3) → (T, 75, 3) float32.

        Matches the offline pipeline: hip-center + shoulder-scale invariance,
        clip to [-10, 10], then z-score with the training mean/std.
        """
        window = _translate_scale_invariant(window)
        window = np.clip(window, -_CLIP, _CLIP)
        return ((window - self._mean[0]) / self._std[0]).astype(np.float32)
