"""Redis-backed session state for WebSocket gesture streaming.

Handles:
- Session creation with reconnect tokens
- State persistence across reconnects
- TTL management (active vs disconnected)
- Atomic counter updates for stats

Key schema
----------
  glossa:gesture:session:{session_id}      → JSON hash (session data)
  glossa:gesture:token:{reconnect_token}   → session_id  (lookup index)

TTLs:
  Active session:       3600s  (1 hour)
  Disconnected session: 300s   (5 minutes, allows reconnect)
"""

from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

import orjson
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from glossa_common.logging import get_logger

logger = get_logger(__name__)

_SESSION_PREFIX = "glossa:gesture:session:"
_TOKEN_PREFIX = "glossa:gesture:token:"
_ACTIVE_TTL = 3600
_DISCONNECTED_TTL = 300


class SessionStatus(StrEnum):
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"


class GestureSessionData(BaseModel):
    session_id: str
    reconnect_token: str
    client_addr: str
    created_at: float           # unix timestamp
    connected_at: float
    disconnected_at: float | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    frames_received: int = 0
    frames_dropped: int = 0
    windows_processed: int = 0
    predictions_emitted: int = 0
    window_size: int = 30
    stride: int = 15
    max_fps: float = 30.0


class SessionNotFoundError(Exception):
    pass


class InvalidReconnectTokenError(Exception):
    pass


class RedisSessionStore:
    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ── Session lifecycle ──────────────────────────────────────────────────────

    async def create(
        self,
        client_addr: str,
        window_size: int = 30,
        stride: int = 15,
        max_fps: float = 30.0,
    ) -> GestureSessionData:
        session_id = str(uuid4())
        reconnect_token = secrets.token_hex(32)
        now = time.time()

        session = GestureSessionData(
            session_id=session_id,
            reconnect_token=reconnect_token,
            client_addr=client_addr,
            created_at=now,
            connected_at=now,
            window_size=window_size,
            stride=stride,
            max_fps=max_fps,
        )

        pipe = self._redis.pipeline(transaction=True)
        pipe.set(
            _SESSION_PREFIX + session_id,
            orjson.dumps(session.model_dump()),
            ex=_ACTIVE_TTL,
        )
        pipe.set(_TOKEN_PREFIX + reconnect_token, session_id, ex=_ACTIVE_TTL)
        await pipe.execute()

        logger.info("session_created", session_id=session_id, client=client_addr)
        return session

    async def get(self, session_id: str) -> GestureSessionData:
        raw = await self._redis.get(_SESSION_PREFIX + session_id)
        if not raw:
            raise SessionNotFoundError(session_id)
        return GestureSessionData.model_validate(orjson.loads(raw))

    async def resume_by_token(
        self, reconnect_token: str, client_addr: str
    ) -> GestureSessionData:
        session_id_bytes = await self._redis.get(_TOKEN_PREFIX + reconnect_token)
        if not session_id_bytes:
            raise InvalidReconnectTokenError("Token expired or invalid")

        session_id = session_id_bytes.decode() if isinstance(session_id_bytes, bytes) else session_id_bytes
        session = await self.get(session_id)

        if session.status == SessionStatus.EXPIRED:
            raise InvalidReconnectTokenError("Session expired")

        now = time.time()
        session.status = SessionStatus.ACTIVE
        session.connected_at = now
        session.disconnected_at = None
        session.client_addr = client_addr

        await self._save(session, ttl=_ACTIVE_TTL)
        logger.info("session_resumed", session_id=session_id, client=client_addr)
        return session

    async def mark_disconnected(self, session_id: str) -> None:
        try:
            session = await self.get(session_id)
        except SessionNotFoundError:
            return

        session.status = SessionStatus.DISCONNECTED
        session.disconnected_at = time.time()
        await self._save(session, ttl=_DISCONNECTED_TTL)

        # Shrink token TTL to match
        await self._redis.expire(_TOKEN_PREFIX + session.reconnect_token, _DISCONNECTED_TTL)
        logger.info("session_disconnected", session_id=session_id)

    async def delete(self, session_id: str) -> None:
        try:
            session = await self.get(session_id)
        except SessionNotFoundError:
            return
        pipe = self._redis.pipeline()
        pipe.delete(_SESSION_PREFIX + session_id)
        pipe.delete(_TOKEN_PREFIX + session.reconnect_token)
        await pipe.execute()
        logger.info("session_deleted", session_id=session_id)

    # ── Atomic stat updates ────────────────────────────────────────────────────

    async def increment(self, session_id: str, **fields: int) -> None:
        """Atomically increment integer fields.  Silently no-ops if session gone."""
        try:
            session = await self.get(session_id)
        except SessionNotFoundError:
            return

        for field_name, delta in fields.items():
            current = getattr(session, field_name, 0)
            setattr(session, field_name, current + delta)

        # Use remaining TTL to avoid resetting it
        ttl = await self._redis.ttl(_SESSION_PREFIX + session_id)
        await self._save(session, ttl=max(ttl, 60))

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _save(self, session: GestureSessionData, ttl: int = _ACTIVE_TTL) -> None:
        await self._redis.set(
            _SESSION_PREFIX + session.session_id,
            orjson.dumps(session.model_dump()),
            ex=ttl,
        )
