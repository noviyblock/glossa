"""WebSocket session registry.

Keeps a live mapping of session_id → WebSocket so the orchestrator
can push results to the right client at any time.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, WebSocket] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────── #

    def register(self, session_id: str, ws: WebSocket) -> None:
        self._sessions[session_id] = ws
        logger.info("Session registered: %s  (active=%d)", session_id, len(self._sessions))

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info("Session removed: %s  (active=%d)", session_id, len(self._sessions))

    # ── Sending ────────────────────────────────────────────────────────────── #

    async def send(self, session_id: str, payload: dict[str, Any]) -> bool:
        """Send a JSON message to a specific session.

        Returns True on success, False if the session is gone.
        """
        ws = self._sessions.get(session_id)
        if ws is None:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception as exc:
            logger.warning("send failed session=%s: %s", session_id, exc)
            self.unregister(session_id)
            return False

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send to all active sessions (used for system-wide notifications)."""
        dead: list[str] = []
        for sid, ws in list(self._sessions.items()):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.unregister(sid)

    # ── Introspection ─────────────────────────────────────────────────────── #

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def is_active(self, session_id: str) -> bool:
        return session_id in self._sessions
