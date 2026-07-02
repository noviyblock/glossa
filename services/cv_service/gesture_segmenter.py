from __future__ import annotations

import logging
from collections import deque

import numpy as np

from config import (
    GESTURE_HAND_PRESENCE_CONF,
    GESTURE_MAX_FRAMES,
    GESTURE_MIN_FRAMES,
    GESTURE_OFFSET_FRAMES,
    GESTURE_OFFSET_THRESHOLD,
    GESTURE_ONSET_FRAMES,
    GESTURE_ONSET_THRESHOLD,
    GESTURE_PREROLL_FRAMES,
    WINDOW_SIZE,
)
from sliding_window import _resample_to

logger = logging.getLogger("gesture_segmenter")

# True anatomical shoulder indices in the extractor's training-order body17
# (indices 0-16 of the 75-pt layout, see keypoint_extractor.py's
# _COCO17_TO_TRAINING_ORDER). NOT the same as normalizer.py's _SHOU_IDX=[5,6]
# (elbow/wrist in anatomical terms, kept there only for bit-for-bit parity
# with the offline training pipeline). This module needs a physically
# meaningful scale reference, so it uses the true shoulders directly.
_SHOULDER_R_IDX = 1
_SHOULDER_L_IDX = 4
_LEFT_HAND  = slice(33, 54)
_RIGHT_HAND = slice(54, 75)


class GestureSegmenter:
    """Per-session IDLE/ACTIVE state machine driven by hand motion.

    Replaces the manual hold-to-sign button: detects gesture onset when
    hand displacement crosses a threshold, offset when hands return to
    stillness, and hands back exactly one accumulated segment per gesture
    (resampled to WINDOW_SIZE) ready for Normalizer + GestureClassifier.
    """

    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id
        self._state = "IDLE"
        self._prev_kp: np.ndarray | None = None
        self._onset_streak = 0
        self._offset_streak = 0
        self._segment: list[np.ndarray] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=GESTURE_PREROLL_FRAMES)
        self._last_activity = 0.0

    # ------------------------------------------------------------------ #

    def push(self, kp: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Feed one raw (75, 3) extractor-output frame (not normalized).

        Returns (window, gesture_active): window is (WINDOW_SIZE, 75, 3)
        float32 ready for Normalizer + GestureClassifier, or None if no
        gesture just completed on this frame.
        """
        kp = kp.astype(np.float32)
        activity, hands_present = self._compute_activity(kp)
        self._last_activity = activity

        if self._state == "IDLE":
            result = self._push_idle(kp, activity, hands_present)
        else:
            result = self._push_active(kp, activity, hands_present)

        self._prev_kp = kp
        return result

    # ------------------------------------------------------------------ #

    def _push_idle(self, kp: np.ndarray, activity: float, hands_present: bool) -> tuple[np.ndarray | None, bool]:
        self._preroll.append(kp)

        if hands_present and activity >= GESTURE_ONSET_THRESHOLD:
            self._onset_streak += 1
        else:
            self._onset_streak = 0

        if self._onset_streak >= GESTURE_ONSET_FRAMES:
            self._state = "ACTIVE"
            self._segment = list(self._preroll)
            self._onset_streak = 0
            self._offset_streak = 0
            logger.info("session=%s IDLE->ACTIVE activity=%.4f preroll_len=%d",
                        self._session_id[:8], activity, len(self._segment))
            return None, True

        return None, False

    def _push_active(self, kp: np.ndarray, activity: float, hands_present: bool) -> tuple[np.ndarray | None, bool]:
        self._segment.append(kp)

        if (not hands_present) or activity <= GESTURE_OFFSET_THRESHOLD:
            self._offset_streak += 1
        else:
            self._offset_streak = 0

        if len(self._segment) >= GESTURE_MAX_FRAMES:
            logger.warning("session=%s ACTIVE force-flush at MAX_FRAMES=%d",
                            self._session_id[:8], GESTURE_MAX_FRAMES)
            window = self._finish_segment()
            self._segment = []
            self._offset_streak = 0
            return window, True

        if self._offset_streak >= GESTURE_OFFSET_FRAMES:
            segment_len = len(self._segment)
            window = self._finish_segment()
            logger.info("session=%s ACTIVE->IDLE segment_len=%d discarded=%s",
                        self._session_id[:8], segment_len, window is None)
            self._state = "IDLE"
            self._segment = []
            self._onset_streak = 0
            self._offset_streak = 0
            self._preroll.clear()
            return window, False

        return None, True

    # ------------------------------------------------------------------ #

    def _finish_segment(self) -> np.ndarray | None:
        if len(self._segment) < GESTURE_MIN_FRAMES:
            return None
        raw = np.stack(self._segment, axis=0)
        return _resample_to(raw, WINDOW_SIZE)

    def _compute_activity(self, kp: np.ndarray) -> tuple[float, bool]:
        """(activity_score, hands_present) for the current raw frame.

        activity_score: max over the two hands of confidence-weighted mean
        keypoint displacement vs the previous frame, normalized by true
        shoulder width (scale-invariant across camera distance). Using max
        rather than pooling both hands avoids diluting one-handed signs.
        """
        left_conf  = float(kp[_LEFT_HAND, 2].mean())
        right_conf = float(kp[_RIGHT_HAND, 2].mean())
        hands_present = max(left_conf, right_conf) >= GESTURE_HAND_PRESENCE_CONF

        if self._prev_kp is None:
            return 0.0, hands_present

        shoulder_dist = float(np.linalg.norm(
            kp[_SHOULDER_L_IDX, :2] - kp[_SHOULDER_R_IDX, :2]
        ))
        if shoulder_dist < 1e-6:
            return 0.0, hands_present

        left_activity  = self._hand_displacement(kp, self._prev_kp, _LEFT_HAND) / shoulder_dist
        right_activity = self._hand_displacement(kp, self._prev_kp, _RIGHT_HAND) / shoulder_dist
        return max(left_activity, right_activity), hands_present

    @staticmethod
    def _hand_displacement(kp: np.ndarray, prev_kp: np.ndarray, hand: slice) -> float:
        cur, prev = kp[hand], prev_kp[hand]
        weight = np.minimum(cur[:, 2], prev[:, 2])
        weight_sum = float(weight.sum())
        if weight_sum < 1e-6:
            return 0.0
        disp = np.linalg.norm(cur[:, :2] - prev[:, :2], axis=1)
        return float((disp * weight).sum() / weight_sum)

    # ------------------------------------------------------------------ #

    def force_reset(self) -> None:
        """Abandon the current segment, return to IDLE without classifying."""
        self._state = "IDLE"
        self._segment = []
        self._onset_streak = 0
        self._offset_streak = 0
        self._preroll.clear()

    def force_flush(self) -> np.ndarray | None:
        """Classify whatever is currently accumulated, then reset."""
        window = self._finish_segment()
        self.force_reset()
        return window

    @property
    def debug_state(self) -> dict:
        return {
            "state": self._state,
            "segment_len": len(self._segment),
            "onset_streak": self._onset_streak,
            "offset_streak": self._offset_streak,
            "last_activity": round(self._last_activity, 4),
        }
