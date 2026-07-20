"""Unit tests for cv_service's _evict_session -- the fix for a real
regression: a session_id reused across unrelated recognition runs (e.g.
video-upload, then live camera, same session_id -- the client only ever
generates one session_id per app lifetime) inherited stale mid-gesture
GestureSegmenter/TrackState from whatever ran before it, since the only
existing cleanup was a 5-minute idle-TTL sweep. See main.py's
POST /reset_session and orchestrator.py's end_recognition_session, called
from the WS "end_session" path.

Run: pytest services/cv_service/tests/ -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("config", None)

import main as cv_main  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_session_dicts():
    """main.py's per-session dicts are module-level globals -- isolate
    each test from whatever an earlier test left behind."""
    for d in (cv_main._session_segmenters, cv_main._session_smoothers,
              cv_main._session_tracks, cv_main._session_locks,
              cv_main._diag_counters, cv_main._session_last_access):
        d.clear()
    yield
    for d in (cv_main._session_segmenters, cv_main._session_smoothers,
              cv_main._session_tracks, cv_main._session_locks,
              cv_main._diag_counters, cv_main._session_last_access):
        d.clear()


def _populate_session(sid: str) -> None:
    cv_main._session_segmenters[sid] = object()
    cv_main._session_smoothers[sid] = object()
    cv_main._session_tracks[sid] = object()
    cv_main._session_locks[sid] = asyncio.Lock()  # unlocked -- real Lock, .locked() must work
    cv_main._diag_counters[sid] = 3
    cv_main._session_last_access[sid] = 123.0


def test_evict_session_removes_all_per_session_state():
    _populate_session("s1")

    evicted = cv_main._evict_session("s1")

    assert evicted is True
    assert "s1" not in cv_main._session_segmenters
    assert "s1" not in cv_main._session_smoothers
    assert "s1" not in cv_main._session_tracks
    assert "s1" not in cv_main._session_locks
    assert "s1" not in cv_main._diag_counters
    assert "s1" not in cv_main._session_last_access


def test_evict_session_unknown_session_is_a_harmless_noop():
    evicted = cv_main._evict_session("never-seen")

    assert evicted is True  # nothing to skip, so it "succeeds" trivially


def test_evict_session_does_not_touch_other_sessions():
    _populate_session("s1")
    _populate_session("s2")

    cv_main._evict_session("s1")

    assert "s1" not in cv_main._session_segmenters
    assert "s2" in cv_main._session_segmenters


@pytest.mark.asyncio
async def test_evict_session_skips_a_session_whose_lock_is_held():
    import asyncio
    _populate_session("s1")
    real_lock = asyncio.Lock()
    await real_lock.acquire()
    cv_main._session_locks["s1"] = real_lock

    evicted = cv_main._evict_session("s1")

    assert evicted is False
    assert "s1" in cv_main._session_segmenters  # left alone, not evicted mid-request
    real_lock.release()


@pytest.mark.asyncio
async def test_reset_session_endpoint_evicts_and_returns_status():
    _populate_session("s1")

    result = await cv_main.reset_session({"session_id": "s1"})

    assert result == {"evicted": True}
    assert "s1" not in cv_main._session_segmenters


@pytest.mark.asyncio
async def test_reset_session_endpoint_requires_session_id():
    response = await cv_main.reset_session({})

    assert response.status_code == 400
