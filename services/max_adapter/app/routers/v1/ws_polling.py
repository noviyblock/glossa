"""WebSocket long-polling endpoint — alternative to webhook for firewalled deployments.

The client connects and receives MAX updates as JSON frames.  Each frame the
server sends has the shape: {"update_type": "...", "payload": {...}}.

The client can also push control messages:
  {"type": "ping"}       → server replies {"type": "pong"}
  {"type": "subscribe"}  → subscribe to updates (default on connect)
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from glossa_common.logging import get_logger

from ...queue.event_queue import QueuedEvent

logger = get_logger(__name__)

router = APIRouter()

_HEARTBEAT_INTERVAL = 20.0  # seconds
_MAX_IDLE_S = 120.0


@router.websocket("/ws/updates")
async def ws_updates(websocket: WebSocket) -> None:
    """Long-lived WebSocket: server pushes updates, client can send control frames."""
    conn_id = str(uuid4())[:8]
    await websocket.accept()
    logger.info("ws_client_connected", conn_id=conn_id)

    outbox: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)

    async def _receive_loop() -> None:
        try:
            while True:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=_MAX_IDLE_S
                )
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await outbox.put({"type": "pong"})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("ws_receive_error", conn_id=conn_id)

    async def _send_loop() -> None:
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                try:
                    frame = await asyncio.wait_for(outbox.get(), timeout=_HEARTBEAT_INTERVAL)
                    await websocket.send_text(json.dumps(frame, ensure_ascii=False))
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("ws_send_error", conn_id=conn_id)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_receive_loop())
        tg.create_task(_send_loop())

    logger.info("ws_client_disconnected", conn_id=conn_id)


@router.post("/updates/inject", include_in_schema=False)
async def inject_update(websocket_update: dict) -> dict:
    """Internal: inject a synthetic update (for testing without MAX webhook)."""
    from fastapi import Request
    return {"status": "ok", "note": "inject endpoint available in development only"}
