"""Unit tests for TokenBucketThrottle."""

import time

import pytest

from app.domain.throttle import TokenBucketThrottle


def test_allows_up_to_burst():
    t = TokenBucketThrottle(max_fps=10.0, burst=3.0)
    results = [t.check() for _ in range(3)]
    assert all(r.allowed for r in results)


def test_blocks_after_burst_exhausted():
    t = TokenBucketThrottle(max_fps=1.0, burst=2.0)
    t.check()
    t.check()
    result = t.check()
    assert not result.allowed
    assert result.tokens_remaining == 0.0


def test_refills_over_time(monkeypatch):
    calls = []
    start = time.monotonic()

    class _FakeMono:
        _t = start
        def monotonic(self):
            return self._t

    fake = _FakeMono()
    monkeypatch.setattr(time, "monotonic", fake.monotonic)

    t = TokenBucketThrottle(max_fps=10.0, burst=1.0)
    t.check()  # consume the 1 token

    # Advance time by 0.1s → should get 1 new token (10fps × 0.1s = 1 token)
    fake._t = start + 0.1
    result = t.check()
    assert result.allowed


def test_update_max_fps():
    t = TokenBucketThrottle(max_fps=30.0, burst=5.0)
    t.update_max_fps(15.0)
    assert t.max_fps == 15.0


def test_suggested_fps_on_block():
    t = TokenBucketThrottle(max_fps=30.0, burst=1.0)
    t.check()
    result = t.check()
    assert not result.allowed
    assert 0 < result.suggested_fps <= 30.0


def test_invalid_fps_raises():
    with pytest.raises(ValueError):
        TokenBucketThrottle(max_fps=0.0)
    with pytest.raises(ValueError):
        TokenBucketThrottle(max_fps=-1.0)
