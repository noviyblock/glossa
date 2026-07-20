"""Unit tests for SlidingWindowBuffer."""

import asyncio

import numpy as np
import pytest

from app.domain.sliding_window import FrameMeta, SlidingWindowBuffer


def _make_feature(val: float = 0.0, dim: int = 8) -> np.ndarray:
    return np.full(dim, val, dtype=np.float32)


def _meta(seq: int) -> FrameMeta:
    return FrameMeta(seq=seq, ts_ms=float(seq * 33), frame_index=seq)


@pytest.mark.asyncio
async def test_no_emit_before_window_full():
    buf = SlidingWindowBuffer(window_size=5, stride=3)
    for i in range(4):
        window, dropped = await buf.push(_make_feature(float(i)), _meta(i))
        assert window is None
        assert not dropped


@pytest.mark.asyncio
async def test_emits_when_window_full_and_stride_met():
    buf = SlidingWindowBuffer(window_size=5, stride=5)
    windows = []
    for i in range(5):
        w, _ = await buf.push(_make_feature(float(i)), _meta(i))
        if w is not None:
            windows.append(w)
    assert len(windows) == 1
    assert windows[0].feature_matrix.shape == (5, 8)
    assert windows[0].start_seq == 0
    assert windows[0].end_seq == 4


@pytest.mark.asyncio
async def test_overlapping_windows_stride_less_than_window():
    buf = SlidingWindowBuffer(window_size=4, stride=2)
    windows = []
    for i in range(8):
        w, _ = await buf.push(_make_feature(float(i)), _meta(i))
        if w is not None:
            windows.append(w)
    # First window at frame 3 (window_size=4, stride=2, emit after 4+2=6 frames? no)
    # stride=2: emit when len>=4 AND frames_since_emit>=2
    # frame 0,1,2,3 → len==4, frames_since_emit==4>=2 → emit
    # frame 4,5     → frames_since_emit==2>=2 → emit
    # frame 6,7     → emit
    assert len(windows) >= 2
    for w in windows:
        assert w.feature_matrix.shape == (4, 8)


@pytest.mark.asyncio
async def test_eviction_when_buffer_full():
    buf = SlidingWindowBuffer(window_size=3, stride=3, max_buffer=5)
    dropped_count = 0
    for i in range(10):
        _, dropped = await buf.push(_make_feature(float(i)), _meta(i))
        if dropped:
            dropped_count += 1
    assert dropped_count > 0
    assert buf.total_dropped > 0


@pytest.mark.asyncio
async def test_reset_clears_buffer():
    buf = SlidingWindowBuffer(window_size=5, stride=5)
    for i in range(4):
        await buf.push(_make_feature(), _meta(i))
    assert buf.depth == 4
    await buf.reset()
    assert buf.depth == 0


@pytest.mark.asyncio
async def test_window_feature_values_correct():
    buf = SlidingWindowBuffer(window_size=3, stride=3)
    for i in range(3):
        w, _ = await buf.push(_make_feature(float(i)), _meta(i))
    assert w is not None
    # Row 0 should be all 0.0, row 1 all 1.0, row 2 all 2.0
    np.testing.assert_allclose(w.feature_matrix[0], np.zeros(8))
    np.testing.assert_allclose(w.feature_matrix[1], np.ones(8))
    np.testing.assert_allclose(w.feature_matrix[2], np.full(8, 2.0))
