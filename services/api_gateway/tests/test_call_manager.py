"""Unit tests for CallManager (two-party call pairing, punch-list plan item
8). Uses the same in-memory _FakeRedis stand-in as the other orchestrator
tests -- CallManager only ever calls setex/get on it.

Run: pytest services/api_gateway/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("config", None)

from call_manager import CallManager  # noqa: E402


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)


@pytest.mark.asyncio
async def test_create_call_returns_a_short_code() -> None:
    calls = CallManager(_FakeRedis())

    call_id = await calls.create_call("host-session")

    assert len(call_id) == 6
    assert call_id == call_id.upper()


@pytest.mark.asyncio
async def test_host_has_no_peer_before_anyone_joins() -> None:
    calls = CallManager(_FakeRedis())
    call_id = await calls.create_call("host-session")

    peer = await calls.get_peer("host-session")

    assert peer is None


@pytest.mark.asyncio
async def test_join_pairs_host_and_guest_both_directions() -> None:
    calls = CallManager(_FakeRedis())
    call_id = await calls.create_call("host-session")

    joined = await calls.join_call(call_id, "guest-session")

    assert joined is True
    assert await calls.get_peer("host-session") == "guest-session"
    assert await calls.get_peer("guest-session") == "host-session"


@pytest.mark.asyncio
async def test_join_unknown_call_id_fails() -> None:
    calls = CallManager(_FakeRedis())

    joined = await calls.join_call("NOSUCH", "guest-session")

    assert joined is False


@pytest.mark.asyncio
async def test_unrelated_session_has_no_peer() -> None:
    calls = CallManager(_FakeRedis())
    call_id = await calls.create_call("host-session")
    await calls.join_call(call_id, "guest-session")

    peer = await calls.get_peer("some-other-session")

    assert peer is None


@pytest.mark.asyncio
async def test_get_peer_on_expired_or_unknown_session_returns_none() -> None:
    calls = CallManager(_FakeRedis())

    peer = await calls.get_peer("never-registered")

    assert peer is None
