"""Tests for session manager and session stores."""

from __future__ import annotations

import pytest

from app.domain.entities import TranslationMode, UserSession
from app.session.session_manager import InMemorySessionStore, SessionManager


@pytest.fixture
def manager() -> SessionManager:
    return SessionManager(InMemorySessionStore(max_sessions=10))


class TestInMemorySessionStore:
    async def test_get_returns_none_on_missing(self) -> None:
        store = InMemorySessionStore()
        result = await store.get(99)
        assert result is None

    async def test_save_and_get_roundtrip(self) -> None:
        store = InMemorySessionStore()
        session = UserSession(user_id=1, chat_id=10)
        await store.save(session)
        loaded = await store.get(1)
        assert loaded is not None
        assert loaded.chat_id == 10

    async def test_lru_eviction(self) -> None:
        store = InMemorySessionStore(max_sessions=3)
        for i in range(4):
            s = UserSession(user_id=i, chat_id=i * 10)
            await store.save(s)
        # First entry (user_id=0) should be evicted
        assert await store.get(0) is None
        assert await store.get(3) is not None

    async def test_delete(self) -> None:
        store = InMemorySessionStore()
        s = UserSession(user_id=5, chat_id=50)
        await store.save(s)
        await store.delete(5)
        assert await store.get(5) is None

    async def test_start_stop_noop(self) -> None:
        store = InMemorySessionStore()
        await store.start()
        await store.stop()


class TestSessionManager:
    async def test_get_or_create_new_session(self, manager: SessionManager) -> None:
        session = await manager.get_or_create(user_id=1, chat_id=10)
        assert session.user_id == 1
        assert session.chat_id == 10
        assert session.mode == TranslationMode.TEXT_RESPONSE

    async def test_get_or_create_returns_existing(self, manager: SessionManager) -> None:
        s1 = await manager.get_or_create(1, 10)
        s1_id = s1.session_id
        s2 = await manager.get_or_create(1, 10)
        assert s2.session_id == s1_id

    async def test_set_mode(self, manager: SessionManager) -> None:
        await manager.get_or_create(1, 10)
        session = await manager.set_mode(1, 10, TranslationMode.SPEECH_TTS)
        assert session.mode == TranslationMode.SPEECH_TTS

        # Persisted
        reloaded = await manager.get_or_create(1, 10)
        assert reloaded.mode == TranslationMode.SPEECH_TTS

    async def test_set_domain(self, manager: SessionManager) -> None:
        await manager.get_or_create(1, 10)
        await manager.set_domain(1, 10, "medical")
        session = await manager.get_or_create(1, 10)
        assert session.domain == "medical"

    async def test_clear_context_resets_conversation(self, manager: SessionManager) -> None:
        session = await manager.get_or_create(1, 10)
        session.add_turn("user", "hello")
        session.add_turn("assistant", "hi")
        await manager.save(session)

        await manager.clear_context(1, 10)
        refreshed = await manager.get_or_create(1, 10)
        assert len(refreshed.conversation) == 0

    async def test_save_increments_total_requests(self, manager: SessionManager) -> None:
        session = await manager.get_or_create(1, 10)
        assert session.total_requests == 0
        await manager.save(session)
        reloaded = await manager.get_or_create(1, 10)
        assert reloaded.total_requests == 1

    async def test_history_text_format(self, manager: SessionManager) -> None:
        session = await manager.get_or_create(1, 10)
        session.add_turn("user", "вопрос")
        session.add_turn("assistant", "ответ")
        history = session.history_text()
        assert "user" in history.lower() or "вопрос" in history

    async def test_add_turn_keeps_last_20(self) -> None:
        session = UserSession(user_id=1, chat_id=10)
        for i in range(25):
            session.add_turn("user", f"msg {i}")
        assert len(session.conversation) == 20
        assert session.conversation[-1].content == "msg 24"

    async def test_different_users_isolated(self, manager: SessionManager) -> None:
        s1 = await manager.get_or_create(1, 10)
        s2 = await manager.get_or_create(2, 20)
        assert s1.session_id != s2.session_id

        await manager.set_mode(1, 10, TranslationMode.GESTURE_TEXT)
        s2_check = await manager.get_or_create(2, 20)
        assert s2_check.mode == TranslationMode.TEXT_RESPONSE
