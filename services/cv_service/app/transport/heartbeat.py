"""WebSocket heartbeat manager — ping/pong with missed-beat detection."""

from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket, WebSocketDisconnect

from glossa_common.logging import get_logger

from .protocol import MsgType, encode_frame, encode_pong

logger = get_logger(__name__)

PING_INTERVAL: float = 15.0    # send ping every N seconds
PONG_TIMEOUT: float = 8.0      # wait N seconds for pong before declaring dead
MAX_MISSED: int = 2             # declare dead after N consecutive missed pongs


class HeartbeatManager:
    """Tracks liveness of a single WebSocket connection.

    Run `start()` as a background task.  Call `on_pong()` whenever a
    PONG frame arrives.  `is_alive` turns False after MAX_MISSED misses.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        ping_interval: float = PING_INTERVAL,
        pong_timeout: float = PONG_TIMEOUT,
        max_missed: int = MAX_MISSED,
    ) -> None:
        self._ws = websocket
        self._session_id = session_id
        self._ping_interval = ping_interval
        self._pong_timeout = pong_timeout
        self._max_missed = max_missed

        self._seq: int = 0
        self._pending_pong: bool = False
        self._last_pong_at: float = time.monotonic()
        self._missed: int = 0
        self._alive: bool = True
        self._pong_event: asyncio.Event = asyncio.Event()

    @property
    def is_alive(self) -> bool:
        return self._alive

    def on_pong(self, seq: int) -> None:
        """Called by the frame dispatcher when a PONG arrives."""
        self._last_pong_at = time.monotonic()
        self._pending_pong = False
        self._missed = 0
        self._pong_event.set()
        logger.debug("pong_received", session_id=self._session_id, seq=seq)

    def on_any_message(self) -> None:
        """Reset liveness on any inbound message (data frames count as alive)."""
        self._last_pong_at = time.monotonic()
        self._missed = 0

    async def start(self) -> None:
        """Long-running coroutine — cancel to stop."""
        while self._alive:
            await asyncio.sleep(self._ping_interval)

            self._seq += 1
            self._pending_pong = True
            self._pong_event.clear()

            try:
                ping_frame = encode_frame(MsgType.PING, b"", seq=self._seq)
                await self._ws.send_bytes(ping_frame)
                logger.debug("ping_sent", session_id=self._session_id, seq=self._seq)
            except Exception:
                self._alive = False
                logger.warning("ping_send_failed", session_id=self._session_id)
                return

            # Wait for pong with timeout
            try:
                await asyncio.wait_for(
                    self._pong_event.wait(), timeout=self._pong_timeout
                )
            except asyncio.TimeoutError:
                self._missed += 1
                logger.warning(
                    "pong_timeout",
                    session_id=self._session_id,
                    seq=self._seq,
                    missed=self._missed,
                )
                if self._missed >= self._max_missed:
                    logger.error(
                        "connection_declared_dead",
                        session_id=self._session_id,
                        missed=self._missed,
                    )
                    self._alive = False
                    return

    def stop(self) -> None:
        self._alive = False
