from __future__ import annotations

import math

import numpy as np

from config import SMOOTH_BETA, SMOOTH_DCUTOFF, SMOOTH_MINCUTOFF


class OneEuroFilter:
    """One Euro Filter (Casiez, Roussel, Vogel — CHI 2012).

    Adaptive low-pass: heavier smoothing when the signal is nearly still
    (kills jitter), lighter smoothing when it's moving fast (avoids lag on
    real motion) — unlike a fixed-alpha EMA, which is always one or the
    other. Timestamp-driven rather than a fixed assumed frame rate, since
    actual round-trip cadence varies with network/inference load.
    """

    def __init__(self, mincutoff: float = SMOOTH_MINCUTOFF, beta: float = SMOOTH_BETA,
                 dcutoff: float = SMOOTH_DCUTOFF) -> None:
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, te: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x, t
            return x

        te = max(t - self._t_prev, 1e-3)  # guard against duplicate/out-of-order timestamps
        dx = (x - self._x_prev) / te
        a_d = self._alpha(self._dcutoff, te)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self._mincutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, te)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class KeypointSmoother:
    """Per-session One Euro smoothing over the x/y channels of a (75, 3) frame.

    Runs 75×2 independent filters (one per joint per axis). A joint with
    score == 0 (hard-zeroed as absent, see keypoint_extractor.py) resets its
    filter pair rather than smoothing through the zero — otherwise a hand
    re-entering frame after being absent would be dragged from (0, 0),
    producing a visible swoop/lag artifact instead of appearing immediately.
    """

    def __init__(self, num_joints: int = 75) -> None:
        self._fx = [OneEuroFilter() for _ in range(num_joints)]
        self._fy = [OneEuroFilter() for _ in range(num_joints)]

    def smooth(self, kp: np.ndarray, t: float) -> np.ndarray:
        out = kp.copy()
        for i in range(kp.shape[0]):
            if kp[i, 2] <= 0.0:
                self._fx[i].reset()
                self._fy[i].reset()
                continue
            out[i, 0] = self._fx[i](float(kp[i, 0]), t)
            out[i, 1] = self._fy[i](float(kp[i, 1]), t)
        return out
