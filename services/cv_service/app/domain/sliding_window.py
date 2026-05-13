"""Sliding window buffer for temporal gesture feature sequences.

Thread-safe via asyncio.Lock.  Emits a numpy window whenever the
stride condition is met; caller decides what to do with it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from glossa_common.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FrameMeta:
    seq: int
    ts_ms: float
    frame_index: int


@dataclass
class WindowReady:
    """Emitted when the buffer has enough frames for a full inference window."""

    feature_matrix: np.ndarray   # shape (window_size, feature_dim)
    frames_meta: list[FrameMeta]
    start_seq: int
    end_seq: int
    buffer_depth: int             # how many frames are currently buffered


class SlidingWindowBuffer:
    """Circular buffer that emits overlapping windows for ONNX inference.

    Parameters
    ----------
    window_size:
        Number of frames in one inference window (e.g. 30).
    stride:
        How many *new* frames must arrive before the next emission.
        stride < window_size → overlap; stride == window_size → no overlap.
    max_buffer:
        Hard cap on stored frames.  Oldest frames are evicted when
        exceeded (backpressure indicator returned to caller).
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 15,
        max_buffer: int = 90,
    ) -> None:
        if stride < 1 or stride > window_size:
            raise ValueError(f"stride must be in [1, {window_size}], got {stride}")

        self._window_size = window_size
        self._stride = stride
        self._max_buffer = max_buffer

        self._frames: deque[np.ndarray] = deque()
        self._meta: deque[FrameMeta] = deque()
        self._frames_since_emit: int = 0
        self._total_received: int = 0
        self._total_dropped: int = 0
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def push(
        self, feature: np.ndarray, meta: FrameMeta
    ) -> tuple[WindowReady | None, bool]:
        """Push one feature vector.

        Returns
        -------
        (window, was_dropped)
        window is non-None when a complete window is ready for inference.
        was_dropped is True when the buffer was full and an old frame evicted.
        """
        async with self._lock:
            return self._push_locked(feature, meta)

    def _push_locked(
        self, feature: np.ndarray, meta: FrameMeta
    ) -> tuple[WindowReady | None, bool]:
        dropped = False

        # Evict oldest if at capacity
        if len(self._frames) >= self._max_buffer:
            self._frames.popleft()
            self._meta.popleft()
            self._total_dropped += 1
            dropped = True
            logger.debug(
                "frame_evicted",
                total_dropped=self._total_dropped,
                buffer_depth=len(self._frames),
            )

        self._frames.append(feature)
        self._meta.append(meta)
        self._frames_since_emit += 1
        self._total_received += 1

        window: WindowReady | None = None

        if (
            len(self._frames) >= self._window_size
            and self._frames_since_emit >= self._stride
        ):
            frames_list = list(self._frames)[-self._window_size :]
            meta_list = list(self._meta)[-self._window_size :]

            window = WindowReady(
                feature_matrix=np.stack(frames_list, axis=0),
                frames_meta=meta_list,
                start_seq=meta_list[0].seq,
                end_seq=meta_list[-1].seq,
                buffer_depth=len(self._frames),
            )
            self._frames_since_emit = 0

        return window, dropped

    async def reset(self) -> None:
        async with self._lock:
            self._frames.clear()
            self._meta.clear()
            self._frames_since_emit = 0

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def depth(self) -> int:
        return len(self._frames)

    @property
    def total_received(self) -> int:
        return self._total_received

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def stride(self) -> int:
        return self._stride
