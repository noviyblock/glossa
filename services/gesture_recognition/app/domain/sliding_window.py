"""Sliding window manager for temporal gesture classification.

Decoupled from any I/O — pure data transformation.

Window strategy
---------------
Given a stream of T frames:
  - A window of size W slides with stride S.
  - When (T - W) % S == 0 a new window is emitted.
  - Windows overlap by (W - S) frames → temporal continuity.
  - Boundary condition: final partial window is padded with last frame.

Deduplication
-------------
Consecutive identical predictions below a merge threshold are merged
into a single token spanning both windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .entities import GlossCandidate, TopKResult


# ── Feature constants ────────────────────────────────────────────────────────
# Must match the feature extractor (MediaPipe → feature_vector)
POSE_N = 33
HAND_N = 21
POSE_DIM = POSE_N * 4      # x y z visibility
HAND_DIM = HAND_N * 3      # x y z  ×2
FEATURE_DIM = POSE_DIM + HAND_DIM * 2   # 132 + 63 + 63 = 258


@dataclass(slots=True)
class FrameRecord:
    seq: int
    ts_ms: float
    features: np.ndarray     # shape (FEATURE_DIM,)


@dataclass
class WindowBatch:
    """One emission from the sliding window."""
    windows: list[np.ndarray]       # each (W, D)
    frame_ranges: list[tuple[int, int]]   # (start_seq, end_seq) per window
    total_frames: int


class SlidingWindowManager:
    """Stateful sliding window — call `push()` per frame, `flush()` at end.

    Parameters
    ----------
    window_size : int
        Frames per window (T). Typically 30 for 1s at 30 fps.
    stride : int
        Frames between window starts. stride < window_size → overlap.
    pad_incomplete : bool
        Pad the final partial window with the last frame instead of dropping.
    feature_dim : int
        Feature vector length per frame.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 15,
        pad_incomplete: bool = True,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1 or stride > window_size:
            raise ValueError(f"stride must be in [1, {window_size}]")

        self.window_size = window_size
        self.stride = stride
        self.pad_incomplete = pad_incomplete
        self.feature_dim = feature_dim

        self._buffer: deque[FrameRecord] = deque()
        self._frames_since_emit: int = 0
        self._total_frames: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, seq: int, ts_ms: float, features: np.ndarray) -> list[np.ndarray] | None:
        """Push one frame.  Returns a list of ready windows (usually 0 or 1)."""
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected features shape ({self.feature_dim},), got {features.shape}"
            )

        self._buffer.append(FrameRecord(seq=seq, ts_ms=ts_ms, features=features))
        self._frames_since_emit += 1
        self._total_frames += 1

        # Trim buffer to window_size (keep last W frames)
        while len(self._buffer) > self.window_size:
            self._buffer.popleft()

        if (
            len(self._buffer) >= self.window_size
            and self._frames_since_emit >= self.stride
        ):
            self._frames_since_emit = 0
            return [self._extract_window()]

        return None

    def flush(self) -> list[np.ndarray]:
        """Call at end of stream to emit any pending partial window."""
        if not self._buffer or not self.pad_incomplete:
            return []

        if len(self._buffer) < self.window_size:
            return [self._pad_window()]

        return []

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_emit = 0
        self._total_frames = 0

    @property
    def buffer_depth(self) -> int:
        return len(self._buffer)

    @property
    def total_frames_seen(self) -> int:
        return self._total_frames

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_window(self) -> np.ndarray:
        """Stack current buffer as [W, D]."""
        return np.stack([r.features for r in self._buffer], axis=0)

    def _pad_window(self) -> np.ndarray:
        """Pad with last frame repetition to reach window_size."""
        frames = list(self._buffer)
        last = frames[-1].features if frames else np.zeros(self.feature_dim, dtype=np.float32)
        pad_n = self.window_size - len(frames)
        padded = [r.features for r in frames] + [last] * pad_n
        return np.stack(padded, axis=0)


# ── Post-processing ───────────────────────────────────────────────────────────

def deduplicate_predictions(
    results: list[TopKResult],
    merge_threshold: float = 0.5,
) -> list[TopKResult]:
    """Merge consecutive identical best-gloss predictions.

    Two adjacent windows are merged if:
    - Same best gloss
    - Both confidences > merge_threshold
    - The combined window is contiguous
    """
    if not results:
        return []

    merged: list[TopKResult] = [results[0]]

    for curr in results[1:]:
        prev = merged[-1]
        prev_best = prev.best
        curr_best = curr.best

        if (
            prev_best is not None
            and curr_best is not None
            and prev_best.gloss == curr_best.gloss
            and prev_best.confidence > merge_threshold
            and curr_best.confidence > merge_threshold
        ):
            # Merge: extend end frame, average confidence
            avg_conf = (prev_best.confidence + curr_best.confidence) / 2.0
            merged_candidates = [
                GlossCandidate(
                    confidence=avg_conf,
                    gloss=prev_best.gloss,
                    gloss_id=prev_best.gloss_id,
                    start_frame=prev_best.start_frame,
                    end_frame=curr_best.end_frame,
                )
            ] + [c for c in curr.candidates[1:]]

            merged[-1] = TopKResult(
                candidates=merged_candidates,
                is_fallback=prev.is_fallback or curr.is_fallback,
                window_start=prev.window_start,
                window_end=curr.window_end,
            )
        else:
            merged.append(curr)

    return merged


def filter_low_confidence(
    results: list[TopKResult],
    threshold: float = 0.35,
) -> list[TopKResult]:
    return [r for r in results if r.best_confidence >= threshold]
