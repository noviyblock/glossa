"""Conversation memory stores — in-memory and Redis-backed."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict

from glossa_common.logging import get_logger

from ..domain.entities import ConversationMemory, ConversationTurn, Domain
from ..domain.interfaces import MemoryStorePort

logger = get_logger(__name__)

_ACTIVE_TTL      = 7200    # 2 hours
_IDLE_TTL        = 600     # 10 minutes (keep short for idle sessions)


class InMemoryConversationStore(MemoryStorePort):
    """Single-instance in-process memory store."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[ConversationMemory, float]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="mem-cleanup")

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()

    async def load(self, session_id: str) -> ConversationMemory | None:
        async with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            memory, expires = entry
            if time.time() > expires:
                del self._store[session_id]
                return None
            return memory

    async def save(self, memory: ConversationMemory) -> None:
        async with self._lock:
            ttl = _ACTIVE_TTL
            self._store[memory.session_id] = (memory, time.time() + ttl)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(120)
            now = time.time()
            async with self._lock:
                expired = [k for k, (_, exp) in self._store.items() if now > exp]
                for k in expired:
                    del self._store[k]
            if expired:
                logger.info("mem_cleanup", evicted=len(expired))


class RedisConversationStore(MemoryStorePort):
    """Redis-backed conversation memory — required for multi-instance."""

    def __init__(self, redis_url: str = "redis://redis:6379/2") -> None:
        self._url = redis_url
        self._redis = None

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            logger.info("redis_conv_store_connected", url=self._url)
        except Exception as exc:
            logger.warning("redis_conv_store_unavailable", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def load(self, session_id: str) -> ConversationMemory | None:
        if not self._redis:
            return None
        raw = await self._redis.get(f"nlp:conv:{session_id}")
        if not raw:
            return None
        try:
            data = json.loads(raw)
            turns = [ConversationTurn(**t) for t in data.pop("turns", [])]
            memory = ConversationMemory(**data, turns=turns)
            return memory
        except Exception:
            logger.exception("redis_conv_load_error", session_id=session_id)
            return None

    async def save(self, memory: ConversationMemory) -> None:
        if not self._redis:
            return
        try:
            data = {
                "session_id": memory.session_id,
                "summary": memory.summary,
                "dominant_domain": memory.dominant_domain,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "turns": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "intent": t.intent,
                        "domain": t.domain,
                    }
                    for t in memory.turns
                ],
            }
            await self._redis.setex(
                f"nlp:conv:{memory.session_id}",
                _ACTIVE_TTL,
                json.dumps(data, ensure_ascii=False),
            )
        except Exception:
            logger.exception("redis_conv_save_error", session_id=memory.session_id)

    async def delete(self, session_id: str) -> None:
        if self._redis:
            await self._redis.delete(f"nlp:conv:{session_id}")
