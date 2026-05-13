"""ASR session store — in-memory with optional Redis backend.

Supports reconnection via a reconnect_token that survives WebSocket drops.
TTL: active sessions kept for 3600 s, disconnected for 300 s.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import ASRSession, ASRSessionStatus
from ..domain.interfaces import SessionStorePort

logger = get_logger(__name__)

_ACTIVE_TTL     = 3600   # 1 hour
_DISCONNECTED_TTL = 300  # 5 minutes


class InMemorySessionStore(SessionStorePort):
    """Thread-safe in-memory session store. Suitable for single-instance deployments."""

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[ASRSession, float]] = {}   # id → (session, expires_at)
        self._tokens: dict[str, str] = {}                           # token → session_id
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="session-cleanup")

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()

    # ── SessionStorePort ──────────────────────────────────────────────────────

    async def create(self, session: ASRSession) -> None:
        async with self._lock:
            expires = time.time() + _ACTIVE_TTL
            self._sessions[session.session_id] = (session, expires)
            self._tokens[session.reconnect_token] = session.session_id
        logger.info("session_created", session_id=session.session_id)

    async def get(self, session_id: str) -> ASRSession | None:
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            session, expires_at = entry
            if time.time() > expires_at:
                await self._evict(session_id)
                return None
            return session

    async def get_by_token(self, token: str) -> ASRSession | None:
        async with self._lock:
            session_id = self._tokens.get(token)
            if session_id is None:
                return None
        return await self.get(session_id)

    async def update(self, session: ASRSession) -> None:
        async with self._lock:
            if session.session_id not in self._sessions:
                return
            ttl = _ACTIVE_TTL if session.status == ASRSessionStatus.ACTIVE else _DISCONNECTED_TTL
            self._sessions[session.session_id] = (session, time.time() + ttl)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            await self._evict(session_id)
        logger.info("session_deleted", session_id=session_id)

    # ── Reconnect helpers ─────────────────────────────────────────────────────

    async def mark_disconnected(self, session_id: str) -> None:
        session = await self.get(session_id)
        if session:
            session.status = ASRSessionStatus.DISCONNECTED
            await self.update(session)
            logger.info("session_disconnected", session_id=session_id)

    async def resume(self, token: str) -> ASRSession | None:
        """Resume a disconnected session by token. Returns None if expired."""
        session = await self.get_by_token(token)
        if session and session.status == ASRSessionStatus.DISCONNECTED:
            session.status = ASRSessionStatus.ACTIVE
            session.touch()
            await self.update(session)
            logger.info("session_resumed", session_id=session.session_id)
            return session
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _evict(self, session_id: str) -> None:
        entry = self._sessions.pop(session_id, None)
        if entry:
            session, _ = entry
            self._tokens.pop(session.reconnect_token, None)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with self._lock:
                expired = [
                    sid for sid, (_, exp) in self._sessions.items() if now > exp
                ]
                for sid in expired:
                    await self._evict(sid)
            if expired:
                logger.info("session_cleanup", evicted=len(expired))


class RedisSessionStore(SessionStorePort):
    """Redis-backed session store — required for multi-instance deployments."""

    def __init__(self, redis_url: str = "redis://redis:6379/1") -> None:
        self._url = redis_url
        self._redis: Any = None

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._url, decode_responses=False)
            await self._redis.ping()
            logger.info("redis_session_store_connected", url=self._url)
        except Exception as exc:
            logger.warning("redis_session_store_unavailable", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def create(self, session: ASRSession) -> None:
        if not self._redis:
            return
        import json, dataclasses
        payload = json.dumps(dataclasses.asdict(session)).encode()
        await self._redis.setex(f"asr:session:{session.session_id}", _ACTIVE_TTL, payload)
        await self._redis.setex(f"asr:token:{session.reconnect_token}", _ACTIVE_TTL, session.session_id.encode())

    async def get(self, session_id: str) -> ASRSession | None:
        if not self._redis:
            return None
        import json
        raw = await self._redis.get(f"asr:session:{session_id}")
        if not raw:
            return None
        data = json.loads(raw)
        return ASRSession(**data)

    async def get_by_token(self, token: str) -> ASRSession | None:
        if not self._redis:
            return None
        sid_bytes = await self._redis.get(f"asr:token:{token}")
        if not sid_bytes:
            return None
        return await self.get(sid_bytes.decode())

    async def update(self, session: ASRSession) -> None:
        if not self._redis:
            return
        import json, dataclasses
        ttl = _ACTIVE_TTL if session.status == ASRSessionStatus.ACTIVE else _DISCONNECTED_TTL
        payload = json.dumps(dataclasses.asdict(session)).encode()
        await self._redis.setex(f"asr:session:{session.session_id}", ttl, payload)

    async def delete(self, session_id: str) -> None:
        if not self._redis:
            return
        session = await self.get(session_id)
        if session:
            await self._redis.delete(f"asr:token:{session.reconnect_token}")
        await self._redis.delete(f"asr:session:{session_id}")
