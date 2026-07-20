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
    translation and records what it was called with. /reset_session
    echoes `reset_session_glosses` (default []), matching cv_service's
    real /reset_session response shape (see cv_service/main.py) -- lets
    tests simulate a gesture that was still ACTIVE and got force-flushed
    when the session ended (see test_end_recognition_session_buffers_a_
    force_flushed_trailing_gesture)."""

    def __init__(
        self, cv_responses: list[dict[str, Any]],
        reset_session_glosses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._cv_responses = list(cv_responses)
        self._reset_session_glosses = reset_session_glosses or []
        self.nlp_calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        if url == f"{CV_SERVICE_URL}/process_frame":
            return _FakeResponse(self._cv_responses.pop(0))
        if url == f"{CV_SERVICE_URL}/reset_session":
            return _FakeResponse({"evicted": True, "glosses": self._reset_session_glosses})
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


def _cv_frame(gloss: str, prob: float, extra: list[tuple[str, float]] | None = None) -> dict[str, Any]:
    # cv_service always returns TOP_K=3 candidates in practice (see
    # gesture_classifier.py) -- `extra` lets tests express a realistic
    # runner-up shape instead of a single-item list, which matters for
    # _gesture_is_confident's margin-based exception (see
    # test_standout_single_candidate_below_threshold_is_still_buffered /
    # test_ambiguous_low_confidence_with_close_runnerup_is_not_buffered).
    candidates = [{"gloss": gloss, "prob": prob}]
    if extra:
        candidates += [{"gloss": g, "prob": p} for g, p in extra]
    return {
        "glosses": candidates,
        "person_detected": True,
        "gesture_active": False,
        "preview": False,
    }


@pytest.mark.asyncio
async def test_low_confidence_gloss_is_not_buffered() -> None:
    """A completed 'gesture' below _MIN_CONFIDENCE, with a close runner-up
    (genuine ambiguity, not a standout) must not enter pending_positions --
    it's noise, not a real classification (same gate the client-side
    display filter now mirrors, see punch-list item 1). The close runner-up
    also matters here: it keeps _gesture_is_confident's margin-based
    exception from kicking in, which is exactly what
    test_ambiguous_low_confidence_with_close_runnerup_is_not_buffered
    checks explicitly."""
    below = _MIN_CONFIDENCE - 0.01
    assert below >= 0.0, "test assumes _MIN_CONFIDENCE > 0"
    http = _CvHTTPClient([_cv_frame("шум", below, extra=[("шум2", below - 0.05), ("шум3", below - 0.10)])])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s1", "fake-frame-b64")

    assert result["translation"] == ""
    assert "pending_positions" not in result  # nothing changed this frame
    session = await orch._get_session("s1")
    assert session["pending_positions"] == []
    assert http.nlp_calls == []  # never even attempted a flush


@pytest.mark.asyncio
async def test_standout_single_candidate_below_threshold_is_still_buffered() -> None:
    """Real regression: a genuine, unambiguous gesture ('память') landed at
    ~0.30 confidence -- below _MIN_CONFIDENCE, but a clear standout over
    both runner-ups -- and was wrongly rejected outright. Normal for a
    200-way classifier, where even a correct top-1 rarely dominates the
    softmax the way it would in a small-class problem.
    _gesture_is_confident's margin path should let this through."""
    http = _CvHTTPClient([_cv_frame("память", 0.30, extra=[("привет", 0.02), ("пока", 0.01)])])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s9", "fake-frame-b64")

    assert result["pending_positions"] == [[
        {"gloss": "память", "prob": 0.30},
        {"gloss": "привет", "prob": 0.02},
        {"gloss": "пока", "prob": 0.01},
    ]]


@pytest.mark.asyncio
async def test_ambiguous_low_confidence_with_close_runnerup_is_not_buffered() -> None:
    """A low top-1 that's NOT a standout (close runner-up) must still be
    rejected -- this is exactly the transitional-hand-movement noise
    _MIN_CONFIDENCE was originally raised to filter out, and the margin
    exception must not accidentally reopen that hole."""
    http = _CvHTTPClient([_cv_frame("шум", 0.28, extra=[("шум2", 0.25)])])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s10", "fake-frame-b64")

    assert "pending_positions" not in result


@pytest.mark.asyncio
async def test_below_margin_floor_is_never_buffered_even_as_sole_candidate() -> None:
    """A very low top-1 (below _MIN_CONFIDENCE_MARGIN_FLOOR) must be
    rejected even as the sole/standout candidate -- the margin exception
    isn't a way to accept near-zero-confidence noise, only a way to accept
    a below-_MIN_CONFIDENCE but still reasonably confident standout."""
    http = _CvHTTPClient([_cv_frame("шум", 0.10)])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s11", "fake-frame-b64")

    assert "pending_positions" not in result


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


@pytest.mark.asyncio
async def test_end_recognition_session_flushes_buffered_but_unsent_sentence() -> None:
    """Real regression: a video-upload's clip ends abruptly (WS
    end_session) well before SENTENCE_PAUSE_SECONDS or MAX_SENTENCE_GLOSSES
    would naturally flush the buffer -- whatever was recognized during the
    upload used to be silently discarded (pending_positions just cleared),
    so the client never got a translated result at all."""
    above = _MIN_CONFIDENCE + 0.5
    http = _CvHTTPClient([_cv_frame("бабочка", above)])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    result = await orch.process_frame("s7", "fake-frame-b64")
    assert result["pending_positions"] == [[{"gloss": "бабочка", "prob": above}]]
    assert http.nlp_calls == []  # not flushed yet -- only one gesture buffered

    flush_result = await orch.end_recognition_session("s7")

    assert flush_result is not None
    assert flush_result["translation"] == "stub translation"
    assert len(http.nlp_calls) == 1
    assert http.nlp_calls[0]["positions"] == [[{"gloss": "бабочка", "prob": above}]]
    session = await orch._get_session("s7")
    assert session["pending_positions"] == []


@pytest.mark.asyncio
async def test_end_recognition_session_with_nothing_pending_is_a_noop() -> None:
    orch = Orchestrator(redis=_FakeRedis(), http=_CvHTTPClient([]))

    flush_result = await orch.end_recognition_session("s8")

    assert flush_result is None


@pytest.mark.asyncio
async def test_end_recognition_session_buffers_a_force_flushed_trailing_gesture() -> None:
    """Real regression: short uploaded clips recognized nothing at all --
    cv_service's segmenter was still ACTIVE (never reached its natural
    offset) when the clip ended, so /reset_session now force-flushes and
    classifies it (see cv_service's docstring) instead of discarding it.
    end_recognition_session must fold that trailing gesture into
    pending_positions (if confident enough) and flush the whole thing,
    not just the gestures that had already completed naturally."""
    above = _MIN_CONFIDENCE + 0.5
    http = _CvHTTPClient([], reset_session_glosses=[{"gloss": "память", "prob": above}])
    orch = Orchestrator(redis=_FakeRedis(), http=http)
    # Nothing buffered yet via process_frame -- the WHOLE clip was one
    # gesture that never closed naturally, only force-flushed on reset.

    flush_result = await orch.end_recognition_session("s12")

    assert flush_result is not None
    assert flush_result["translation"] == "stub translation"
    assert len(http.nlp_calls) == 1
    assert http.nlp_calls[0]["positions"] == [[{"gloss": "память", "prob": above}]]


@pytest.mark.asyncio
async def test_end_recognition_session_drops_a_low_confidence_trailing_gesture() -> None:
    """A force-flushed trailing gesture still has to clear
    _gesture_is_confident, same as any other gesture -- reset_session
    shouldn't become a backdoor around the confidence gate."""
    http = _CvHTTPClient([], reset_session_glosses=[
        {"gloss": "шум", "prob": 0.10},
    ])
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    flush_result = await orch.end_recognition_session("s13")

    assert flush_result is None
    assert http.nlp_calls == []


@pytest.mark.asyncio
async def test_end_recognition_session_combines_pending_and_trailing_gesture() -> None:
    """A sentence with some gestures already buffered normally PLUS one
    more that only closed via force-flush on reset -- both must end up in
    the same translated sentence, not just the trailing one."""
    above = _MIN_CONFIDENCE + 0.5
    http = _CvHTTPClient(
        [_cv_frame("привет", above)],
        reset_session_glosses=[{"gloss": "пока", "prob": above}],
    )
    orch = Orchestrator(redis=_FakeRedis(), http=http)

    await orch.process_frame("s14", "fake-frame-b64")
    flush_result = await orch.end_recognition_session("s14")

    assert flush_result is not None
    assert http.nlp_calls[0]["positions"] == [
        [{"gloss": "привет", "prob": above}],
        [{"gloss": "пока", "prob": above}],
    ]
