"""WebSocket ASR streaming endpoint.

Protocol documentation:
  ws://host/api/v1/ws/asr?language=ru&sample_rate=16000&emit_partials=true

  After connecting, the server immediately sends SESSION_INFO (binary frame, type=0x14).
  Client should extract session_id and reconnect_token for reconnect support.

  To reconnect after disconnect:
    Send binary frame: type=0x05 (RECONNECT), JSON payload: {"reconnect_token": "<token>"}

  Binary frames (header: 8 bytes — type:1, flags:1, seq:2, length:4 big-endian):
    0x01  AUDIO_CHUNK   — raw PCM int16 payload (16 kHz default)
    0x02  CONTROL       — JSON: {"language": "en"} or {"sample_rate": 16000}
    0x03  PING          — empty payload, seq echoed as PONG
    0x04  END_OF_STREAM — flush buffer and emit final transcript
    0x05  RECONNECT     — JSON: {"reconnect_token": "<token>"}

  Server emits:
    0x10  PARTIAL       — JSON partial transcript
    0x11  FINAL         — JSON final transcript
    0x12  PONG          — heartbeat response
    0x13  ERROR         — JSON error
    0x14  SESSION_INFO  — JSON session metadata
    0x15  BACKPRESSURE  — JSON slow-down signal
"""

from __future__ import annotations

from fastapi import APIRouter, Query, WebSocket

from ...dependencies import get_audio_queue, get_medical_adapter, get_session_store, get_transcriber, get_vad
from ...transport.websocket_handler import ASRWebSocketHandler

router = APIRouter(tags=["asr-websocket"])


@router.websocket("/api/v1/ws/asr")
async def asr_websocket(
    ws: WebSocket,
    language: str = Query(default="ru", description="BCP-47 language code"),
    sample_rate: int = Query(default=16000, ge=8000, le=48000),
    emit_partials: bool = Query(default=True),
    beam_size: int = Query(default=5, ge=1, le=20),
    word_timestamps: bool = Query(default=False),
    transcriber=get_transcriber,
    vad=get_vad,
    medical=get_medical_adapter,
    audio_queue=get_audio_queue,
    session_store=get_session_store,
):
    """Binary WebSocket endpoint for real-time streaming ASR.

    Accepts raw PCM int16 audio chunks and emits partial + final transcripts.
    Supports reconnection via reconnect_token.
    """
    handler = ASRWebSocketHandler(
        ws=ws,
        transcriber=ws.app.state.transcriber,
        vad=ws.app.state.vad,
        medical=ws.app.state.medical_adapter,
        queue=ws.app.state.audio_queue,
        session_store=ws.app.state.session_store,
        language=language,
        sample_rate=sample_rate,
        emit_partials=emit_partials,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
    )
    await handler.run()
