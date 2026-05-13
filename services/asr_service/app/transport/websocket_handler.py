"""WebSocket handler — drives the ASR pipeline per connection.

Responsibilities:
- Handshake: create/resume session, send SESSION_INFO
- Binary frames → AudioChunk → queue.submit()
- Heartbeat: PING → PONG
- Control frames: language/sample_rate reconfiguration
- END_OF_STREAM: flush pipeline, send FINAL
- Graceful disconnect: mark session disconnected, preserve state
- Reconnect: resume via reconnect_token
- Backpressure: send BACKPRESSURE when queue is full
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from glossa_common.logging import get_logger

from ..domain.entities import (
    ASRSession,
    ASRSessionStatus,
    AudioChunk,
    AudioEncoding,
    FinalTranscript,
    PartialTranscript,
)
from ..domain.interfaces import MedicalAdapterPort, TranscriberPort, VADPort
from ..pipeline.asr_pipeline import ASRPipeline
from ..queue.audio_queue import AudioChunkQueue
from ..session.asr_session import InMemorySessionStore
from .protocol import (
    MsgType,
    ProtocolError,
    decode_frame,
    encode_backpressure,
    encode_error,
    encode_final,
    encode_partial,
    encode_pong,
    encode_session_info,
)

logger = get_logger(__name__)

_HEARTBEAT_INTERVAL = 20.0   # seconds
_HEARTBEAT_TIMEOUT  = 60.0
_MAX_MISSED_PINGS   = 3


class ASRWebSocketHandler:
    """Manages one WebSocket connection lifecycle."""

    def __init__(
        self,
        ws: WebSocket,
        transcriber: TranscriberPort,
        vad: VADPort,
        medical: MedicalAdapterPort,
        queue: AudioChunkQueue,
        session_store: InMemorySessionStore,
        language: str = "ru",
        sample_rate: int = 16000,
        emit_partials: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = False,
    ) -> None:
        self._ws = ws
        self._transcriber = transcriber
        self._vad = vad
        self._medical = medical
        self._queue = queue
        self._store = session_store

        self._language = language
        self._sample_rate = sample_rate
        self._emit_partials = emit_partials
        self._beam_size = beam_size
        self._word_timestamps = word_timestamps

        self._session: ASRSession | None = None
        self._pipeline: ASRPipeline | None = None
        self._send_lock = asyncio.Lock()
        self._seq = 0
        self._last_pong = time.monotonic()
        self._missed_pings = 0

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ws.accept()
        try:
            await self._handshake()
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._receive_loop(), name="ws-receive")
                tg.create_task(self._heartbeat_loop(), name="ws-heartbeat")
        except* WebSocketDisconnect:
            pass
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception("ws_handler_error", exc=str(exc))
        finally:
            await self._disconnect()

    # ── Handshake ─────────────────────────────────────────────────────────────

    async def _handshake(self) -> None:
        """Create new session or resume disconnected one."""
        session = ASRSession(
            language=self._language,
            sample_rate=self._sample_rate,
        )
        await self._store.create(session)
        self._session = session
        self._pipeline = self._build_pipeline(session)
        self._queue.register_handler(session.session_id, self._pipeline.push_chunk)

        await self._send_raw(encode_session_info(
            session.session_id,
            session.reconnect_token,
            session.language,
        ))
        logger.info("ws_session_created", session_id=session.session_id)

    async def _try_resume(self, token: str) -> bool:
        resumed = await self._store.resume(token)
        if resumed is None:
            return False
        self._session = resumed
        self._pipeline = self._build_pipeline(resumed)
        self._queue.register_handler(resumed.session_id, self._pipeline.push_chunk)
        await self._send_raw(encode_session_info(
            resumed.session_id,
            resumed.reconnect_token,
            resumed.language,
        ))
        logger.info("ws_session_resumed", session_id=resumed.session_id)
        return True

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        chunk_seq = 0
        while True:
            try:
                raw = await self._ws.receive_bytes()
            except WebSocketDisconnect:
                return

            try:
                frame = decode_frame(raw)
            except ProtocolError as exc:
                await self._send_raw(encode_error(exc.code, str(exc)))
                continue

            if frame.msg_type == MsgType.AUDIO_CHUNK:
                await self._handle_audio(frame.payload, chunk_seq)
                chunk_seq += 1

            elif frame.msg_type == MsgType.PING:
                self._last_pong = time.monotonic()
                self._missed_pings = 0
                await self._send_raw(encode_pong(frame.seq))

            elif frame.msg_type == MsgType.CONTROL:
                await self._handle_control(frame.json())

            elif frame.msg_type == MsgType.END_OF_STREAM:
                await self._handle_eos()

            elif frame.msg_type == MsgType.RECONNECT:
                data = frame.json()
                if not await self._try_resume(data.get("reconnect_token", "")):
                    await self._send_raw(encode_error("SESSION_EXPIRED", "Session not found or expired"))

    # ── Audio handling ────────────────────────────────────────────────────────

    async def _handle_audio(self, payload: bytes, seq: int) -> None:
        if self._session is None:
            return

        chunk = AudioChunk(
            data=payload,
            sample_rate=self._sample_rate,
            encoding=AudioEncoding.PCM_S16LE,
            session_id=self._session.session_id,
            seq=seq,
        )
        accepted = self._queue.submit(chunk)
        if not accepted:
            await self._send_raw(encode_backpressure(
                reason="queue_full",
                queue_depth=self._queue.depth,
                suggested_action="reduce_send_rate",
            ))

    async def _handle_control(self, data: dict) -> None:
        if "language" in data:
            self._language = data["language"]
            if self._session:
                self._session.language = self._language
                await self._store.update(self._session)
        if "sample_rate" in data:
            self._sample_rate = int(data["sample_rate"])
        logger.info("ws_control_updated", language=self._language, sample_rate=self._sample_rate)

    async def _handle_eos(self) -> None:
        """Flush pipeline and send final transcript."""
        if self._pipeline is None:
            return
        logger.info("ws_end_of_stream", session_id=self._session.session_id if self._session else "")
        await self._pipeline.flush()

    # ── Callbacks from pipeline ───────────────────────────────────────────────

    async def _on_partial(self, partial: PartialTranscript) -> None:
        if not self._emit_partials:
            return
        self._seq += 1
        await self._send_raw(encode_partial(
            text=partial.text,
            language=partial.language,
            seq=self._seq,
            chunk_seq=partial.chunk_seq,
        ))

    async def _on_final(self, final: FinalTranscript) -> None:
        self._seq += 1
        words = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in final.words
        ]
        await self._send_raw(encode_final(
            text=final.text,
            language=final.language,
            language_probability=final.language_probability,
            seq=self._seq,
            inference_ms=final.inference_ms,
            total_latency_ms=final.total_latency_ms,
            corrected_text=final.corrected_text,
            medical_terms=final.medical_terms_found,
            words=words,
        ))

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            elapsed = time.monotonic() - self._last_pong
            if elapsed > _HEARTBEAT_TIMEOUT:
                self._missed_pings += 1
                if self._missed_pings >= _MAX_MISSED_PINGS:
                    logger.warning("ws_heartbeat_timeout", session_id=self._session_id)
                    await self._ws.close(code=1001)
                    return

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _send_raw(self, data: bytes) -> None:
        async with self._send_lock:
            try:
                await self._ws.send_bytes(data)
            except Exception as exc:
                logger.debug("ws_send_failed", error=str(exc))

    async def _disconnect(self) -> None:
        if self._session:
            self._queue.unregister_handler(self._session.session_id)
            await self._store.mark_disconnected(self._session.session_id)
        if self._pipeline:
            await self._pipeline.reset()
        logger.info("ws_disconnected", session_id=self._session_id)

    def _build_pipeline(self, session: ASRSession) -> ASRPipeline:
        return ASRPipeline(
            session=session,
            transcriber=self._transcriber,
            vad=self._vad,
            medical_adapter=self._medical,
            on_partial=self._on_partial,
            on_final=self._on_final,
            emit_partials=self._emit_partials,
            language=self._language,
            beam_size=self._beam_size,
            word_timestamps=self._word_timestamps,
        )

    @property
    def _session_id(self) -> str:
        return self._session.session_id if self._session else "none"
