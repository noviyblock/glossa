"""Integration test: translate_sync degrades gracefully instead of raising
when a downstream service (nlp-service) is unreachable.

Exercises the real Orchestrator.translate_sync() against fake HTTP/Redis
stand-ins (no mocking of translate_sync itself) -- verifies the resilience
refactor in orchestrator.py (shared _translate_reverse/_synthesize_audio/
_render_sign_video helpers, reused from the WS path's process_audio) holds
for the REST text_to_rsl path, which previously let NLP/TTS exceptions
propagate unhandled and fail the whole request.

Run: pytest services/api_gateway/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# services/api_gateway has no __init__.py (flat-file service, not a
# package) -- pytest's default rootdir insertion only adds this file's own
# directory to sys.path, not the parent, so `import orchestrator` needs an
# explicit path insert. Same pattern as scripts/measure_accuracy.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NLP_SERVICE_URL, TTS_SERVICE_URL  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _NlpDownHTTPClient:
    """nlp-service unreachable; asr/tts-service respond normally."""

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        if url.startswith(NLP_SERVICE_URL):
            raise ConnectionError("nlp-service unreachable (simulated)")
        if url == f"{TTS_SERVICE_URL}/synthesize":
            return _FakeResponse({"audio": "ZmFrZS13YXY="})
        if url == f"{TTS_SERVICE_URL}/sign_video":
            return _FakeResponse({"video": None})
        raise AssertionError(f"unexpected call in this test: POST {url}")


class _AllDownHTTPClient:
    """Every downstream call fails."""

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        raise ConnectionError("simulated total outage")


class _FakeRedis:
    """Minimal in-memory stand-in for the handful of redis-py calls
    Orchestrator actually makes -- avoids depending on fakeredis's async API
    surface for a single test file."""

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


@pytest.mark.asyncio
async def test_translate_sync_text_to_rsl_survives_nlp_outage() -> None:
    orch = Orchestrator(redis=_FakeRedis(), http=_NlpDownHTTPClient())

    result = await orch.translate_sync(
        "text_to_rsl", text="Привет, как дела?", session_id="test-session",
    )

    # NLP /translate_reverse failed -> _translate_reverse() echoes the input
    # text instead of the exception propagating out of translate_sync.
    assert result["translation"] == "Привет, как дела?"
    # TTS /synthesize succeeded independently of the NLP failure.
    assert result["audio_wav"] == "ZmFrZS13YXY="
    assert result["video_mp4"] is None
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_translate_sync_text_to_rsl_all_downstream_down() -> None:
    """Even NLP *and* TTS both down -> still returns a dict, never raises."""
    orch = Orchestrator(redis=_FakeRedis(), http=_AllDownHTTPClient())

    result = await orch.translate_sync(
        "text_to_rsl", text="Тест", session_id="test-session-2",
    )

    assert result["translation"] == "Тест"   # echoed, not raised
    assert result["audio_wav"] == ""
    assert result["video_mp4"] is None
