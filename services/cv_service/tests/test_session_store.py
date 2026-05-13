"""Integration tests for RedisSessionStore using fakeredis."""

import pytest
import fakeredis.aioredis as fakeredis

from app.session.redis_session import (
    GestureSessionData,
    InvalidReconnectTokenError,
    RedisSessionStore,
    SessionNotFoundError,
    SessionStatus,
)


@pytest.fixture
async def store() -> RedisSessionStore:
    redis = fakeredis.FakeRedis(decode_responses=False)
    return RedisSessionStore(redis)


@pytest.mark.asyncio
async def test_create_and_get(store: RedisSessionStore):
    session = await store.create(client_addr="127.0.0.1:1234")
    assert session.session_id
    assert session.reconnect_token
    assert session.status == SessionStatus.ACTIVE

    fetched = await store.get(session.session_id)
    assert fetched.session_id == session.session_id
    assert fetched.reconnect_token == session.reconnect_token


@pytest.mark.asyncio
async def test_get_nonexistent_raises(store: RedisSessionStore):
    with pytest.raises(SessionNotFoundError):
        await store.get("nonexistent-id")


@pytest.mark.asyncio
async def test_mark_disconnected(store: RedisSessionStore):
    session = await store.create(client_addr="127.0.0.1:9999")
    await store.mark_disconnected(session.session_id)
    fetched = await store.get(session.session_id)
    assert fetched.status == SessionStatus.DISCONNECTED
    assert fetched.disconnected_at is not None


@pytest.mark.asyncio
async def test_resume_by_valid_token(store: RedisSessionStore):
    session = await store.create(client_addr="127.0.0.1:1111")
    await store.mark_disconnected(session.session_id)

    resumed = await store.resume_by_token(session.reconnect_token, client_addr="127.0.0.1:2222")
    assert resumed.session_id == session.session_id
    assert resumed.status == SessionStatus.ACTIVE
    assert resumed.client_addr == "127.0.0.1:2222"


@pytest.mark.asyncio
async def test_resume_by_invalid_token_raises(store: RedisSessionStore):
    with pytest.raises(InvalidReconnectTokenError):
        await store.resume_by_token("bad-token-xxx", client_addr="127.0.0.1:1")


@pytest.mark.asyncio
async def test_increment_stats(store: RedisSessionStore):
    session = await store.create(client_addr="127.0.0.1:42")
    await store.increment(session.session_id, frames_received=10, predictions_emitted=2)
    fetched = await store.get(session.session_id)
    assert fetched.frames_received == 10
    assert fetched.predictions_emitted == 2


@pytest.mark.asyncio
async def test_delete_removes_session(store: RedisSessionStore):
    session = await store.create(client_addr="127.0.0.1:5")
    await store.delete(session.session_id)
    with pytest.raises(SessionNotFoundError):
        await store.get(session.session_id)


@pytest.mark.asyncio
async def test_create_with_custom_params(store: RedisSessionStore):
    session = await store.create(
        client_addr="10.0.0.1:8888",
        window_size=60,
        stride=30,
        max_fps=15.0,
    )
    assert session.window_size == 60
    assert session.stride == 30
    assert session.max_fps == 15.0
