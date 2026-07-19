"""Unit tests for Orchestrator's sentence-accumulation logic in process_frame:
the _MIN_CONFIDENCE buffering gate, the MAX_SENTENCE_GLOSSES cap, the
SENTENCE_PAUSE_SECONDS timeout flush, and delete_last_pending -- none of
which had test coverage before (see punch-list plan, item 6.2). Uses the
same fake HTTP/Redis stand-in pattern as test_orchestrator_resilience.py.

Run: pytest services/api_gateway/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# See test_orchestrator_resilience.py for why this pop is needed when this
# service's tests run alongside another service's in the same pytest process.
sys.modules.pop("config", None)

from config import CV_SERVICE_URL, MAX_SENTENCE_GLOSSES, NLP_SERVICE_URL  # noqa: E402
from orchestrator import Orchestrator, _MIN_CONFIDENCE  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _CvHTTPClient:
    """Returns canned CV /process_frame responses in call order; NLP
    /translate_sequence_topk (sentence flush) always echoes a fixed stub
    translation and records what it was called with."""

    def __init__(self, cv_responses: list[dict[str, Any]]) -> None:
        self._cv_responses = list(cv_responses)
        self.nlp_calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        if url == f"{CV_SERVICE_URL}/process_frame":
            return _FakeResponse(self._cv_responses.pop(0))
        if url == f"{NLP_SERVICE_URL}/translate_sequence_topk":
            self.nlp_calls.append(json or {})
            return _FakeResponse({"translation": "stub translation"})
        raise AssertionError(f"unexpected call in this test: POST {url}")


class _FakeRedis:
    """Same minimal in-memory stand-in as test_orchestrator_resilience.py."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def xadd(self, *args: Any, **kwargs: Any) -> None:
        pass


def _cv_frame(gloss: str, prob: float) -> dict[str, Any]:
    return {
        "glosses": [{"gloss": gloss, "prob": prob}],
        "person_detected": True,
        "gesture_active": False,
        "preview": False,
    }


@pytest.mark.asyncio
async def test_low_confidence_gloss_is_not_buffered() -> None:
    """A completed 'gesture' below _MIN_CONFIDENCE must not enter
    pending_positions -- it's noise, not a real classification (same gate
    the client-side display filter now mirrors, see punch-list item 1)."""
    below = _MIN_CONFIDENCE - 0.01
    assert below >= 0.0, "test assumes _MIN_CONFIDENCE > 0"
    http = _CvHTTPClient([_cv_frame("шум", below)])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s1", "fake-frame-b64")

    assert result["translation"] == ""
    assert "pending_positions" not in result  # nothing changed this frame
    session = await orch._get_session("s1")
    assert session["pending_positions"] == []
    assert http.nlp_calls == []  # never even attempted a flush


@pytest.mark.asyncio
async def test_high_confidence_gloss_is_buffered() -> None:
    above = _MIN_CONFIDENCE + 0.5
    http = _CvHTTPClient([_cv_frame("привет", above)])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s2", "fake-frame-b64")

    assert result["translation"] == ""  # buffered, not flushed yet
    assert result["pending_positions"] == [[{"gloss": "привет", "prob": above}]]
    session = await orch._get_session("s2")
    assert len(session["pending_positions"]) == 1


@pytest.mark.asyncio
async def test_max_sentence_glosses_triggers_immediate_flush() -> None:
    above = _MIN_CONFIDENCE + 0.5
    frames = [_cv_frame(f"слово{i}", above) for i in range(MAX_SENTENCE_GLOSSES)]
    http = _CvHTTPClient(frames)
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = None
    for _ in range(MAX_SENTENCE_GLOSSES):
        result = await orch.process_frame("s3", "fake-frame-b64")

    assert result["translation"] == "stub translation"
    assert result["pending_positions"] == []  # cleared after flush
    assert len(http.nlp_calls) == 1
    assert len(http.nlp_calls[0]["positions"]) == MAX_SENTENCE_GLOSSES
    session = await orch._get_session("s3")
    assert session["pending_positions"] == []


@pytest.mark.asyncio
async def test_pause_timeout_flushes_buffered_sentence() -> None:
    """No new gesture, but enough wall-clock time has passed since the last
    one -> the buffer flushes on the fallback pause path, not the
    MAX_SENTENCE_GLOSSES path."""
    above = _MIN_CONFIDENCE + 0.5
    empty_frame = {
        "glosses": [], "person_detected": False,
        "gesture_active": False, "preview": False,
    }
    http = _CvHTTPClient([_cv_frame("одиноко", above), empty_frame])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    with patch("orchestrator.time.time", return_value=1_000_000.0):
        first = await orch.process_frame("s4", "fake-frame-b64")
    assert first["translation"] == ""
    assert http.nlp_calls == []

    # Jump well past SENTENCE_PAUSE_SECONDS with no new gesture this frame.
    with patch("orchestrator.time.time", return_value=1_000_100.0):
        second = await orch.process_frame("s4", "fake-frame-b64")

    assert second["translation"] == "stub translation"
    assert second["pending_positions"] == []
    assert len(http.nlp_calls) == 1


@pytest.mark.asyncio
async def test_delete_last_pending_removes_only_the_last_entry() -> None:
    above = _MIN_CONFIDENCE + 0.5
    http = _CvHTTPClient([_cv_frame("первый", above), _cv_frame("второй", above)])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    await orch.process_frame("s5", "fake-frame-b64")
    await orch.process_frame("s5", "fake-frame-b64")
    session = await orch._get_session("s5")
    assert len(session["pending_positions"]) == 2

    remaining = await orch.delete_last_pending("s5")

    assert len(remaining) == 1
    assert remaining[0][0]["gloss"] == "первый"
    session = await orch._get_session("s5")
    assert session["pending_positions"] == remaining


@pytest.mark.asyncio
async def test_delete_last_pending_on_empty_buffer_is_a_noop() -> None:
    orch = Orchestrator(redis=_FakeRedis(), http=_CvHTTPClient([]))

    remaining = await orch.delete_last_pending("s6")

    assert remaining == []
