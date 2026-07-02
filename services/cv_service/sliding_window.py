from __future__ import annotations

from collections import deque

import numpy as np

from config import CONF_THRESHOLD, EARLY_EMIT, WINDOW_SIZE, WINDOW_STRIDE


def _resample_to(window: np.ndarray, target_T: int) -> np.ndarray:
    """Linearly resample temporal axis from T to target_T frames.

    Used to feed the ST-GCN model (trained on 64-frame windows, WINDOW_SIZE)
    from a partially-filled buffer so that first results arrive sooner.
    """
    T, N, C = window.shape
    if T == target_T:
        return window
    src_idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    alpha = (src_idx - lo)[:, None, None]
    return ((1 - alpha) * window[lo] + alpha * window[hi]).astype(np.float32)


class SlidingWindowBuffer:
    """Accumulate per-frame keypoints and emit windows for classification.

    Rules:
    - Emit a window once buf_len >= early_emit AND frames_since_emit >= stride.
    - If the buffer isn't full yet, the current snapshot is resampled to
      window_size so the model always receives exactly the expected shape.
    - Early emit when top-1 confidence >= CONF_THRESHOLD (then reset).
    - Reset after 3 consecutive top-1 gesture changes (rapid scene change).

    With the defaults (EARLY_EMIT=10, WINDOW_STRIDE=5) and ~90ms/frame inference
    (rtmlib RTMDet-nano + RTMW):
      • First result arrives in ~10 × 90ms ≈ 900ms (resampled to WINDOW_SIZE=64).
      • Subsequent results every ~5 × 90ms ≈ 450ms until the buffer fills to 64,
        then every 5 real frames.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
        conf_threshold: float = CONF_THRESHOLD,
        early_emit: int = EARLY_EMIT,
    ) -> None:
        self._window_size    = window_size
        self._stride         = stride
        self._conf_threshold = conf_threshold
        self._early_emit     = early_emit
        self._buf: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_emit = 0
        self._last_gloss: str | None = None
        self._gesture_change_streak = 0

    # ------------------------------------------------------------------ #

    def push(self, kp: np.ndarray) -> np.ndarray | None:
        """Append a (75, 3) keypoint frame.

        Returns a (window_size, 75, 3) window ready for inference when the
        stride and early-emit conditions are met, otherwise None.
        """
        self._buf.append(kp.astype(np.float32))
        self._frames_since_emit += 1

        current_len = len(self._buf)

        # Not enough frames yet
        if current_len < self._early_emit:
            return None
        if self._frames_since_emit < self._stride:
            return None

        self._frames_since_emit = 0
        window = np.stack(list(self._buf), axis=0)  # (current_len, 75, 3)

        # Resample to model input size when buffer isn't full yet
        if current_len < self._window_size:
            window = _resample_to(window, self._window_size)

        return window

    def on_result(self, top1_gloss: str, top1_prob: float) -> bool:
        """Update state after receiving classification results.

        Returns True if the buffer was reset.
        """
        if self._last_gloss is not None and top1_gloss != self._last_gloss:
            self._gesture_change_streak += 1
        else:
            self._gesture_change_streak = 0
        self._last_gloss = top1_gloss

        if top1_prob >= self._conf_threshold or self._gesture_change_streak >= 3:
            self._reset()
            return True
        return False

    def _reset(self) -> None:
        self._buf.clear()
        self._frames_since_emit = 0
        self._gesture_change_streak = 0
        self._last_gloss = None
