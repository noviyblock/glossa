from __future__ import annotations

import numpy as np

from config import NORM_STATS_PATH


class Normalizer:
    def __init__(self, stats_path: str = NORM_STATS_PATH) -> None:
        data = np.load(stats_path)
        # shape (1, 32, 75, 3) → squeeze to (32, 75, 3) then broadcast per-frame
        self._mean = data["mean"].squeeze(0).astype(np.float32)  # (32, 75, 3)
        self._std  = data["std"].squeeze(0).astype(np.float32)   # (32, 75, 3)
        self._std  = np.where(self._std < 1e-6, 1.0, self._std)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Normalise window of shape (T, 75, 3) → (T, 75, 3) float32."""
        return ((window - self._mean) / self._std).astype(np.float32)
