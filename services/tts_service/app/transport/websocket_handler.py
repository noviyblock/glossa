"""WebSocket TTS streaming handler.

Protocol (client → server):
  1. Connect to WS /api/v1/ws/synthesize
  2. Send JSON text frame (SynthesizeWSRequest):
     {"text": "...", "voice_id": "aidar", "format": "wav", "sample_rate": 24000}
  3. Receive binary frames: [8-byte header][wav_bytes_for_one_sentence]
     Header: !BBHi = type(1) flags(1) seq(2) length(4)
     type=0x01 (AUDIO) or 0x02 (FINAL) or 0x03 (ERROR)
     flags bit 0: LAST_CHUNK
     flags bit 1: FROM_CACHE
  4. FINAL frame (type=0x02) or LAST_CHUNK flag signals end of stream.
  5. Send PING (JSON {"type":"ping"}) → receive PONG JSON.

Multiple synthesis requests per connection are supported (sequential).
"""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from glossa_common.logging import get_logger

from ..audio.encoder import FrameFlags, FrameType, pack_frame
from ..domain.entities import AudioFormat, SynthesisRequest
from ..pipeline.tts_pipeline import TTSPipeline

logger = get_logger(__name__)

_HEARTBEAT_INTERVAL_S = 20.0
_SYNTHESIS_TIMEOUT_S  = 60.0


class TTSWebSocketHandler:
    """Handles a single WebSocket connection lifecycle."""

    def __init__(self, pipeline: TTSPipeline) -> None:
        self._pipeline = pipeline

    async def run(self, ws: WebSocket) -> None:
        conn_id = str(uuid4())[:8]
        await ws.accept()
        logger.info("ws_tts_connected", conn_id=conn_id)
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._receive_loop(ws, conn_id))
                tg.create_task(self._heartbeat_loop(ws, conn_id))
        except* WebSocketDisconnect:
            logger.info("ws_tts_disconnected", conn_id=conn_id)
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception("ws_tts_error", conn_id=conn_id, error=str(exc))
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # ── Loops ──────────────────────────────────────────────────────────────────

    async def _receive_loop(self, ws: WebSocket, conn_id: str) -> None:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                raise

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(ws, "invalid_json")
                continue

            msg_type = msg.get("type", "synthesize")
            if msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
                continue

            # Synthesis request
            await self._handle_synthesize(ws, msg, conn_id)

    async def _heartbeat_loop(self, ws: WebSocket, conn_id: str) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            try:
                await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
            except Exception:
                raise WebSocketDisconnect(1001, "heartbeat_failed")

    # ── Synthesis ──────────────────────────────────────────────────────────────

    async def _handle_synthesize(
        self, ws: WebSocket, msg: dict, conn_id: str
    ) -> None:
        try:
            req = SynthesisRequest(
                text=msg.get("text", ""),
                voice_id=msg.get("voice_id", "aidar"),
                language=msg.get("language", "ru"),
                sample_rate=int(msg.get("sample_rate", 24000)),
                format=AudioFormat(msg.get("format", "wav")),
                put_accent=bool(msg.get("put_accent", True)),
                put_yo=bool(msg.get("put_yo", True)),
                session_id=msg.get("session_id", str(uuid4())),
            )
        except (ValueError, KeyError) as exc:
            await self._send_error(ws, f"invalid_request: {exc}")
            return

        if not req.text.strip():
            await self._send_error(ws, "empty_text")
            return

        logger.debug(
            "ws_tts_synthesis_start",
            conn_id=conn_id,
            chars=len(req.text),
            voice=req.voice_id,
        )
        t0 = time.perf_counter()
        chunk_count = 0

        try:
            async for chunk in self._pipeline.synthesize_streaming(req):
                flags = FrameFlags.FROM_CACHE if chunk.chunk_index == 0 else FrameFlags.NONE
                if chunk.is_final:
                    flags |= FrameFlags.LAST_CHUNK

                frame = pack_frame(
                    payload=chunk.audio_bytes,
                    frame_type=FrameType.AUDIO,
                    flags=flags,
                    seq=chunk.chunk_index,
                )
                await ws.send_bytes(frame)
                chunk_count += 1

            # Send FINAL control frame (empty payload)
            final_frame = pack_frame(
                payload=b"",
                frame_type=FrameType.FINAL,
                flags=FrameFlags.NONE,
                seq=chunk_count,
            )
            await ws.send_bytes(final_frame)

        except asyncio.TimeoutError:
            await self._send_error(ws, "synthesis_timeout")
        except Exception as exc:
            logger.exception("ws_tts_synthesis_error", conn_id=conn_id)
            await self._send_error(ws, f"synthesis_error: {exc}")
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ws_tts_synthesis_done",
            conn_id=conn_id,
            chars=len(req.text),
            chunks=chunk_count,
            elapsed_ms=round(elapsed_ms, 1),
        )

    async def _send_error(self, ws: WebSocket, reason: str) -> None:
        try:
            error_frame = pack_frame(
                payload=reason.encode(),
                frame_type=FrameType.ERROR,
                flags=FrameFlags.NONE,
            )
            await ws.send_bytes(error_frame)
        except Exception:
            pass
