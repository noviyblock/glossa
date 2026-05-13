import asyncio
from uuid import UUID, uuid4

import orjson
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from glossa_common.logging import get_logger
from glossa_common.schemas.translation import TranslationMode, TranslationRequest, WebSocketMessage
from glossa_common.telemetry import ACTIVE_WEBSOCKETS

from ...dependencies import get_publisher, get_redis
from ...infrastructure.orchestrator import TranslationOrchestrator

router = APIRouter()
logger = get_logger(__name__)

PING_INTERVAL = 20.0  # seconds


@router.websocket("/ws/translate/{mode}")
async def websocket_translate(
    websocket: WebSocket,
    mode: str,
) -> None:
    if mode not in ("rsl_to_text", "text_to_rsl"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    session_id = uuid4()
    translation_mode = TranslationMode(mode)

    await websocket.accept()
    ACTIVE_WEBSOCKETS.labels(service="api-gateway").inc()
    logger.info("ws_connected", session_id=str(session_id), mode=mode)

    http_client = websocket.app.state.http_client
    settings = websocket.app.state.settings if hasattr(websocket.app.state, "settings") else None

    try:
        ping_task = asyncio.create_task(_ping_loop(websocket, session_id))

        # Send session start ack
        await _send_message(websocket, "session_start", session_id, {"mode": mode})

        from ...config import get_settings
        cfg = get_settings()
        orchestrator = TranslationOrchestrator(http_client, cfg)

        while True:
            raw = await websocket.receive_bytes()
            if not raw:
                continue

            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                await _send_error(websocket, session_id, "INVALID_JSON", "Could not parse message")
                continue

            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                audio_b64 = msg.get("audio")
                if not audio_b64:
                    continue
                import base64
                audio_bytes = base64.b64decode(audio_b64)
                request = TranslationRequest(
                    session_id=session_id,
                    mode=translation_mode,
                    domain=msg.get("domain", "general"),
                )
                async for chunk in orchestrator.stream(request, audio_bytes):
                    await _send_message(websocket, "chunk", session_id, chunk.model_dump(mode="json"))

            elif msg_type == "video_frame":
                frame_data = msg.get("frame")
                if not frame_data:
                    continue
                request = TranslationRequest(
                    session_id=session_id,
                    mode=translation_mode,
                    domain=msg.get("domain", "general"),
                )
                async for chunk in orchestrator.stream_video(request, frame_data):
                    await _send_message(websocket, "chunk", session_id, chunk.model_dump(mode="json"))

            elif msg_type == "end_session":
                await _send_message(websocket, "session_end", session_id, {})
                break

            elif msg_type == "ping":
                await _send_message(websocket, "pong", session_id, {})

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=str(session_id))
    except Exception:
        logger.exception("ws_error", session_id=str(session_id))
        await _send_error(websocket, session_id, "INTERNAL_ERROR", "Unexpected server error")
    finally:
        ping_task.cancel()
        ACTIVE_WEBSOCKETS.labels(service="api-gateway").dec()


async def _ping_loop(ws: WebSocket, session_id: UUID) -> None:
    while True:
        await asyncio.sleep(PING_INTERVAL)
        try:
            await _send_message(ws, "ping", session_id, {})
        except Exception:
            break


async def _send_message(ws: WebSocket, msg_type: str, session_id: UUID, payload: dict) -> None:
    msg = WebSocketMessage(type=msg_type, session_id=session_id, payload=payload)
    await ws.send_bytes(orjson.dumps(msg.model_dump(mode="json")))


async def _send_error(ws: WebSocket, session_id: UUID, code: str, message: str) -> None:
    await _send_message(ws, "error", session_id, {"code": code, "message": message})
