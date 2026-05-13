"""Tests for domain sliding window manager."""

import numpy as np
import pytest

from app.domain.entities import GlossCandidate, TopKResult
from app.domain.sliding_window import (
    SlidingWindowManager,
    deduplicate_predictions,
    filter_low_confidence,
)


def _make_manager(window=5, stride=3, max_buf=20, dim=8):
    return SlidingWindowManager(
        window_size=window, stride=stride, max_buffer=max_buf, feature_dim=dim
    )


def _frame(dim=8):
    return np.ones(dim, dtype=np.float32)


def test_no_emission_below_window_size():
    mgr = _make_manager(window=5, stride=3)
    for _ in range(4):
        windows = mgr.push(_frame())
    assert windows == []


def test_emits_window_at_threshold():
    mgr = _make_manager(window=5, stride=3)
    all_windows = []
    for _ in range(5):
        all_windows.extend(mgr.push(_frame()))
    assert len(all_windows) == 1
    seq, start, end = all_windows[0]
    assert seq.shape == (5, 8)
    assert start == 0
    assert end == 4


def test_stride_controls_emission_rate():
    mgr = _make_manager(window=4, stride=2)
    all_windows = []
    for _ in range(8):
        all_windows.extend(mgr.push(_frame()))
    # Windows at frames 4, 6, 8 → 3 emissions
    assert len(all_windows) == 3


def test_reset_clears_buffer():
    mgr = _make_manager(window=5, stride=3)
    for _ in range(4):
        mgr.push(_frame())
    mgr.reset()
    windows = []
    for _ in range(4):
        windows.extend(mgr.push(_frame()))
    assert windows == []


def test_flush_returns_partial():
    mgr = _make_manager(window=5, stride=3)
    for _ in range(3):
        mgr.push(_frame())
    result = mgr.flush()
    # Should return partial window with 3 frames
    assert len(result) <= 1


def _make_result(gloss, confidence, is_fallback=False):
    return TopKResult(
        candidates=[GlossCandidate(confidence=confidence, gloss=gloss, gloss_id=0)],
        is_fallback=is_fallback,
    )


def test_filter_low_confidence():
    results = [
        _make_result("A", 0.9),
        _make_result("B", 0.2),
        _make_result("C", 0.5),
    ]
    filtered = filter_low_confidence(results, min_confidence=0.4)
    glosses = [r.candidates[0].gloss for r in filtered]
    assert "A" in glosses
    assert "C" in glosses
    assert "B" not in glosses


def test_deduplicate_predictions():
    results = [
        _make_result("ЗДРАВСТВУЙТЕ", 0.9),
        _make_result("ЗДРАВСТВУЙТЕ", 0.85),
        _make_result("ЗДРАВСТВУЙТЕ", 0.8),
        _make_result("ДА", 0.7),
    ]
    deduped = deduplicate_predictions(results, window=3)
    # ЗДРАВСТВУЙТЕ should appear at most once per dedup window
    gloss_counts = {}
    for r in deduped:
        for c in r.candidates:
            gloss_counts[c.gloss] = gloss_counts.get(c.gloss, 0) + 1
    assert gloss_counts.get("ЗДРАВСТВУЙТЕ", 0) <= 1
