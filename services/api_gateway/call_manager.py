"""Two-party live call pairing (MVP — see punch-list plan item 8).

Deliberately thin: reuses both existing one-directional pipelines
(rsl_to_text / text_to_rsl) and both existing session/WS infrastructure
(Orchestrator's gw:session:{session_id}, SessionManager) as-is. This module
adds only the missing piece — associating two independent session_ids with
each other — so main.py's WS handler can relay a finished translation to
the *other* participant instead of only echoing it back to the sender.

Not a room system: exactly two slots (host/guest) per call, no broadcast,
no reconnection handling beyond normal SESSION_TTL/CALL_TTL expiry. See the
plan's "Явно вне объёма" section for what's deliberately not covered.
"""
from __future__ import annotations

import json
import logging
import secrets

import redis.asyncio as aioredis

from config import CALL_TTL

logger = logging.getLogger(__name__)


class CallManager:
    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------ #

    async def create_call(self, host_session_id: str) -> str:
        """Start a new call as the host. Returns a short code (not a UUID —
        a real person needs to read/dictate it to the other participant)."""
        call_id = secrets.token_hex(3).upper()  # e.g. "A1B2C3"
        await self._redis.setex(
            f"gw:call:{call_id}", CALL_TTL,
            json.dumps({"host": host_session_id, "guest": None}),
        )
        await self._redis.setex(f"gw:call_of_session:{host_session_id}", CALL_TTL, call_id)
        logger.info("Call created: call_id=%s host=%s", call_id, host_session_id)
        return call_id

    async def join_call(self, call_id: str, guest_session_id: str) -> bool:
        """Fill the second slot. False if the call doesn't exist/expired."""
        raw = await self._redis.get(f"gw:call:{call_id}")
        if raw is None:
            return False
        data = json.loads(raw)
        data["guest"] = guest_session_id
        await self._redis.setex(f"gw:call:{call_id}", CALL_TTL, json.dumps(data))
        await self._redis.setex(f"gw:call_of_session:{guest_session_id}", CALL_TTL, call_id)
        logger.info("Call joined: call_id=%s guest=%s", call_id, guest_session_id)
        return True

    async def get_peer(self, session_id: str) -> str | None:
        """The other participant's session_id, or None if this session
        isn't in a call (yet), the call has no second participant yet, or
        the call/session pairing has expired."""
        call_id = await self._redis.get(f"gw:call_of_session:{session_id}")
        if call_id is None:
            return None
        raw = await self._redis.get(f"gw:call:{call_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        host, guest = data.get("host"), data.get("guest")
        if session_id == host:
            return guest
        if session_id == guest:
            return host
        return None
