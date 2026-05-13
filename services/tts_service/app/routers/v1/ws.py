"""WebSocket synthesis endpoint."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ...transport.websocket_handler import TTSWebSocketHandler

router = APIRouter()


@router.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket) -> None:
    """
    WebSocket TTS streaming endpoint.

    Protocol:
    - Send JSON: {"text": "...", "voice_id": "aidar", "format": "wav", "sample_rate": 24000}
    - Receive binary frames: 8-byte header + WAV chunk per sentence
    - FINAL frame (type=0x02) signals end of stream
    - Send {"type": "ping"} → receive {"type": "pong", "ts": ...}
    """
    pipeline = websocket.app.state.pipeline
    handler = TTSWebSocketHandler(pipeline)
    await handler.run(websocket)
