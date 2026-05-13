"""FastAPI WebSocket route + dependency injection wiring.

This module is the thin "glue" between FastAPI and the handler.
It owns zero business logic — it only resolves dependencies and
delegates to GestureWebSocketHandler.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from glossa_common.logging import get_logger

from ...inference.queue import InferenceQueue
from ...session.redis_session import RedisSessionStore
from ...transport.websocket_handler import GestureWebSocketHandler
from ...config import CVServiceSettings, get_settings

router = APIRouter()
logger = get_logger(__name__)


# ── Dependency resolvers ──────────────────────────────────────────────────────

def _get_session_store(request: Request) -> RedisSessionStore:
    return request.app.state.session_store  # type: ignore[no-any-return]


def _get_inference_queue(request: Request) -> InferenceQueue:
    return request.app.state.inference_queue  # type: ignore[no-any-return]


def _get_cv_settings() -> CVServiceSettings:
    return get_settings()


# ── Route ─────────────────────────────────────────────────────────────────────

@router.websocket("/ws/gesture")
async def gesture_stream(
    websocket: WebSocket,
    request: Request,
) -> None:
    """WebSocket endpoint for real-time gesture recognition.

    Protocol
    --------
    Binary frames (see transport/protocol.py for header layout):
      CLIENT → SERVER:
        0x01  KEYFRAME       — binary keypoints (pose + hands)
        0x02  PING           — heartbeat request
        0x03  SESSION_RESUME — resume after reconnect (JSON: {reconnect_token})
        0x04  CONTROL        — runtime config (JSON: {max_fps, window_size, stride})

      SERVER → CLIENT:
        0x10  PREDICTION     — gloss result (JSON)
        0x11  PONG           — heartbeat reply
        0x12  BACKPRESSURE   — slow down signal (JSON: {suggested_fps, queue_depth})
        0x13  ERROR          — error detail (JSON)
        0x14  SESSION_CREATED — session info + reconnect token (JSON)
        0x15  ACK            — frame acknowledged

    JSON fallback
    -------------
    Set flags byte to 0x01 (IS_JSON) to send JSON keypoints instead of binary.
    See decode_keyframe_json() in protocol.py for the expected schema.

    Reconnect
    ---------
    On disconnect, the server preserves session state for 5 minutes.
    Reconnect by sending a SESSION_RESUME frame as the first message.
    """
    await websocket.accept()

    session_store: RedisSessionStore = request.app.state.session_store
    inference_queue: InferenceQueue = request.app.state.inference_queue
    settings: CVServiceSettings = get_settings()

    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else 0
    client_addr = f"{client_host}:{client_port}"

    handler = GestureWebSocketHandler(
        websocket=websocket,
        session_store=session_store,
        inference_queue=inference_queue,
        client_addr=client_addr,
        default_window_size=settings.sequence_length,
        default_stride=settings.sequence_stride,
        default_max_fps=30.0,
    )

    logger.info("ws_connection_accepted", client=client_addr)
    await handler.run()
    logger.info("ws_connection_closed", client=client_addr)
