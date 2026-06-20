from __future__ import annotations

from collections import deque

import numpy as np

from config import CONF_THRESHOLD, WINDOW_SIZE, WINDOW_STRIDE


class SlidingWindowBuffer:
    """Accumulate per-frame keypoints and emit windows for classification.

    Rules:
    - Emit a window every WINDOW_STRIDE frames once the buffer is full.
    - Early emit when top-1 confidence >= CONF_THRESHOLD (then reset).
    - Reset after 3 consecutive top-1 gesture changes (rapid scene change).
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> None:
        self._window_size    = window_size
        self._stride         = stride
        self._conf_threshold = conf_threshold
        self._buf: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_emit = 0
        self._last_gloss: str | None = None
        self._gesture_change_streak = 0

    # ------------------------------------------------------------------ #

    def push(self, kp: np.ndarray) -> np.ndarray | None:
        """Append a (75, 3) keypoint frame.

        Returns a (32, 75, 3) window ready for inference when the stride
        condition is met and the buffer is full, otherwise None.
        """
        self._buf.append(kp.astype(np.float32))
        self._frames_since_emit += 1

        if len(self._buf) < self._window_size:
            return None
        if self._frames_since_emit < self._stride:
            return None

        self._frames_since_emit = 0
        return np.stack(list(self._buf), axis=0)  # (T, 75, 3)

    def on_result(self, top1_gloss: str, top1_prob: float) -> bool:
        """Update state after receiving classification results.

        Returns True if the buffer was reset (early emit / gesture change
        streak), which means the caller should also restart its own flow.
        """
        # Track consecutive gesture changes
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
