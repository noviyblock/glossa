"""Redis-backed user session manager with in-memory fallback.

Session key: max:session:{user_id}
TTL: 24h (reset on every interaction)

Session stores:
- current translation mode
- domain preference (general/medical/banking)
- language
- last N conversation turns (for NLP context)
- total request count
- session_id (UUID for pipeline correlation)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict

from glossa_common.logging import get_logger

from ..domain.entities import ConversationTurn, TranslationMode, UserSession
from ..domain.interfaces import SessionStorePort

logger = get_logger(__name__)

_SESSION_KEY = "max:session:{user_id}"
_SESSION_TTL = 86400   # 24 hours


# ── In-memory fallback ────────────────────────────────────────────────────────

class InMemorySessionStore(SessionStorePort):
    def __init__(self, max_sessions: int = 10_000) -> None:
        self._store: OrderedDict[int, UserSession] = OrderedDict()
        self._max = max_sessions
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def get(self, user_id: int) -> UserSession | None:
        async with self._lock:
            session = self._store.get(user_id)
            if session is None:
                return None
            self._store.move_to_end(user_id)
            return session

    async def save(self, session: UserSession) -> None:
        async with self._lock:
            if user_id := session.user_id:
                if user_id not in self._store and len(self._store) >= self._max:
                    self._store.popitem(last=False)
                self._store[user_id] = session
                self._store.move_to_end(user_id)

    async def delete(self, user_id: int) -> None:
        async with self._lock:
            self._store.pop(user_id, None)


# ── Redis-backed ──────────────────────────────────────────────────────────────

class RedisSessionStore(SessionStorePort):
    def __init__(self, redis_url: str = "redis://redis:6379/3", ttl: int = _SESSION_TTL) -> None:
        self._url = redis_url
        self._ttl = ttl
        self._redis = None

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            logger.info("redis_session_store_connected", url=self._url)
        except Exception as exc:
            logger.warning("redis_session_store_unavailable", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def get(self, user_id: int) -> UserSession | None:
        if not self._redis:
            return None
        key = _SESSION_KEY.format(user_id=user_id)
        try:
            raw = await self._redis.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            session = UserSession(
                user_id=data["user_id"],
                chat_id=data["chat_id"],
                mode=TranslationMode(data.get("mode", TranslationMode.TEXT_RESPONSE)),
                language=data.get("language", "ru"),
                domain=data.get("domain", "general"),
                conversation=[
                    ConversationTurn(
                        role=t["role"],
                        content=t["content"],
                        timestamp=t.get("timestamp", time.time()),
                        mode=TranslationMode(t.get("mode", TranslationMode.TEXT_RESPONSE)),
                    )
                    for t in data.get("conversation", [])
                ],
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
                total_requests=data.get("total_requests", 0),
                session_id=data.get("session_id", ""),
            )
            await self._redis.expire(key, self._ttl)
            return session
        except Exception:
            logger.exception("redis_session_get_error", user_id=user_id)
            return None

    async def save(self, session: UserSession) -> None:
        if not self._redis:
            return
        key = _SESSION_KEY.format(user_id=session.user_id)
        try:
            data = json.dumps({
                "user_id": session.user_id,
                "chat_id": session.chat_id,
                "mode": session.mode,
                "language": session.language,
                "domain": session.domain,
                "conversation": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "mode": t.mode,
                    }
                    for t in session.conversation[-20:]
                ],
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "total_requests": session.total_requests,
                "session_id": session.session_id,
            }, ensure_ascii=False)
            await self._redis.setex(key, self._ttl, data)
        except Exception:
            logger.exception("redis_session_save_error", user_id=session.user_id)

    async def delete(self, user_id: int) -> None:
        if self._redis:
            await self._redis.delete(_SESSION_KEY.format(user_id=user_id))


# ── Tiered (memory L1 + Redis L2) ────────────────────────────────────────────

class TieredSessionStore(SessionStorePort):
    def __init__(self, l1: InMemorySessionStore, l2: RedisSessionStore) -> None:
        self._l1 = l1
        self._l2 = l2

    async def start(self) -> None:
        await self._l1.start()
        await self._l2.start()

    async def stop(self) -> None:
        await self._l1.stop()
        await self._l2.stop()

    async def get(self, user_id: int) -> UserSession | None:
        session = await self._l1.get(user_id)
        if session:
            return session
        session = await self._l2.get(user_id)
        if session:
            await self._l1.save(session)
        return session

    async def save(self, session: UserSession) -> None:
        await asyncio.gather(
            self._l1.save(session),
            self._l2.save(session),
        )

    async def delete(self, user_id: int) -> None:
        await asyncio.gather(
            self._l1.delete(user_id),
            self._l2.delete(user_id),
        )


class SessionManager:
    """High-level session facade with get-or-create semantics."""

    def __init__(self, store: SessionStorePort) -> None:
        self._store = store

    async def get_or_create(self, user_id: int, chat_id: int) -> UserSession:
        session = await self._store.get(user_id)
        if session is None:
            session = UserSession(user_id=user_id, chat_id=chat_id)
            await self._store.save(session)
            logger.info("session_created", user_id=user_id, chat_id=chat_id)
        return session

    async def save(self, session: UserSession) -> None:
        session.updated_at = time.time()
        session.total_requests += 1
        await self._store.save(session)

    async def set_mode(self, user_id: int, chat_id: int, mode: TranslationMode) -> UserSession:
        session = await self.get_or_create(user_id, chat_id)
        session.mode = mode
        await self.save(session)
        logger.info("session_mode_changed", user_id=user_id, mode=mode)
        return session

    async def set_domain(self, user_id: int, chat_id: int, domain: str) -> UserSession:
        session = await self.get_or_create(user_id, chat_id)
        session.domain = domain
        await self.save(session)
        return session

    async def clear_context(self, user_id: int, chat_id: int) -> None:
        session = await self.get_or_create(user_id, chat_id)
        session.conversation.clear()
        await self.save(session)
        logger.info("session_context_cleared", user_id=user_id)

    async def delete(self, user_id: int) -> None:
        await self._store.delete(user_id)
