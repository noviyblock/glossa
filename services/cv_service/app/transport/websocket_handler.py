"""WebSocket connection handler — transport layer.

Responsibilities (ONLY):
- Accept / close WebSocket connections
- Decode incoming binary/JSON frames via protocol.py
- Dispatch decoded payloads to the domain layer
- Send encoded responses back to the client
- Run heartbeat manager as a background task
- Forward tracing context per connection

Does NOT contain business logic.  All ML, buffering, and state
management is delegated to the domain / inference / session layers.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import orjson
from fastapi import WebSocket, WebSocketDisconnect
from opentelemetry import trace
from structlog.contextvars import bind_contextvars, clear_contextvars

from glossa_common.logging import get_logger
from glossa_common.telemetry import ACTIVE_WEBSOCKETS

from ..domain.pipeline import frame_to_feature_vector
from ..domain.sliding_window import FrameMeta, SlidingWindowBuffer
from ..domain.throttle import TokenBucketThrottle
from ..inference.queue import InferenceQueue, InferenceTask
from ..session.redis_session import (
    GestureSessionData,
    InvalidReconnectTokenError,
    RedisSessionStore,
    SessionNotFoundError,
)
from .heartbeat import HeartbeatManager
from .protocol import (
    BackpressurePayload,
    ControlPayload,
    FrameFlags,
    MsgType,
    ParsedFrame,
    PredictionPayload,
    SessionCreatedPayload,
    decode_frame,
    decode_keyframe_binary,
    decode_keyframe_json,
    encode_ack,
    encode_backpressure,
    encode_error,
    encode_pong,
    encode_prediction,
    encode_session_created,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Output queue capacity per session
_OUTPUT_QUEUE_SIZE = 64
# Maximum concurrent sessions (DoS protection)
_MAX_SESSIONS_WARN = 500


class GestureWebSocketHandler:
    """Stateful handler for one WebSocket connection.

    Wired together by the FastAPI route.  Instantiated per connection.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_store: RedisSessionStore,
        inference_queue: InferenceQueue,
        client_addr: str,
        default_window_size: int = 30,
        default_stride: int = 15,
        default_max_fps: float = 30.0,
    ) -> None:
        self._ws = websocket
        self._store = session_store
        self._infer_q = inference_queue
        self._client_addr = client_addr
        self._default_window_size = default_window_size
        self._default_stride = default_stride
        self._default_max_fps = default_max_fps

        self._session: GestureSessionData | None = None
        self._buffer: SlidingWindowBuffer | None = None
        self._throttle: TokenBucketThrottle | None = None
        self._output_q: asyncio.Queue = asyncio.Queue(maxsize=_OUTPUT_QUEUE_SIZE)
        self._heartbeat: HeartbeatManager | None = None
        self._trace_id: str = str(uuid4()).replace("-", "")

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Full lifecycle of one WebSocket connection."""
        clear_contextvars()
        bind_contextvars(trace_id=self._trace_id, client=self._client_addr)

        with tracer.start_as_current_span("ws.gesture_session") as span:
            span.set_attribute("client.address", self._client_addr)

            ACTIVE_WEBSOCKETS.labels(service="cv-service").inc()
            try:
                await self._run_inner(span)
            except WebSocketDisconnect as exc:
                logger.info("ws_client_disconnected", code=exc.code)
            except Exception:
                logger.exception("ws_unexpected_error")
                try:
                    await self._ws.send_bytes(encode_error("INTERNAL_ERROR", "Unexpected server error"))
                except Exception:
                    pass
            finally:
                await self._teardown()
                ACTIVE_WEBSOCKETS.labels(service="cv-service").dec()

    # ── Inner loop ────────────────────────────────────────────────────────────

    async def _run_inner(self, span: trace.Span) -> None:
        # Receive the very first frame: must be PING, SESSION_RESUME, or KEYFRAME
        # First message determines whether we create a new session or resume one
        first_raw = await asyncio.wait_for(self._ws.receive_bytes(), timeout=10.0)
        first_frame = decode_frame(first_raw)

        if first_frame.msg_type == MsgType.SESSION_RESUME:
            await self._handle_resume(first_frame)
        else:
            await self._handle_new_session(first_frame)

        session = self._session
        assert session is not None

        bind_contextvars(session_id=session.session_id)
        span.set_attribute("session.id", session.session_id)
        logger.info("session_active", session_id=session.session_id, status=session.status)

        # Start background coroutines
        heartbeat_task = asyncio.create_task(
            self._heartbeat.start(), name=f"heartbeat-{session.session_id}"  # type: ignore[union-attr]
        )
        sender_task = asyncio.create_task(
            self._result_sender(), name=f"sender-{session.session_id}"
        )

        try:
            await self._receive_loop()
        finally:
            heartbeat_task.cancel()
            sender_task.cancel()
            await asyncio.gather(heartbeat_task, sender_task, return_exceptions=True)

    # ── Session setup ─────────────────────────────────────────────────────────

    async def _handle_new_session(self, first_frame: ParsedFrame) -> None:
        session = await self._store.create(
            client_addr=self._client_addr,
            window_size=self._default_window_size,
            stride=self._default_stride,
            max_fps=self._default_max_fps,
        )
        await self._init_session_state(session)

        ack = encode_session_created(
            SessionCreatedPayload(
                session_id=session.session_id,
                reconnect_token=session.reconnect_token,
                window_size=session.window_size,
                stride=session.stride,
                max_fps=session.max_fps,
            )
        )
        await self._ws.send_bytes(ack)

        # Process the first frame if it's a keyframe
        if first_frame.msg_type == MsgType.KEYFRAME:
            await self._handle_keyframe(first_frame)

    async def _handle_resume(self, frame: ParsedFrame) -> None:
        try:
            data = frame.json()
            token = data["reconnect_token"]
        except (KeyError, Exception) as exc:
            await self._ws.send_bytes(encode_error("BAD_RESUME", "Missing reconnect_token"))
            raise WebSocketDisconnect(code=1008) from exc

        try:
            session = await self._store.resume_by_token(token, self._client_addr)
        except (InvalidReconnectTokenError, SessionNotFoundError) as exc:
            await self._ws.send_bytes(encode_error("INVALID_TOKEN", str(exc)))
            raise WebSocketDisconnect(code=1008) from exc

        await self._init_session_state(session)
        ack = encode_session_created(
            SessionCreatedPayload(
                session_id=session.session_id,
                reconnect_token=session.reconnect_token,
                window_size=session.window_size,
                stride=session.stride,
                max_fps=session.max_fps,
            )
        )
        await self._ws.send_bytes(ack)

    def _init_session_state(self, session: GestureSessionData) -> None:
        self._session = session
        self._buffer = SlidingWindowBuffer(
            window_size=session.window_size,
            stride=session.stride,
            max_buffer=session.window_size * 3,
        )
        self._throttle = TokenBucketThrottle(max_fps=session.max_fps, burst=5.0)
        self._heartbeat = HeartbeatManager(
            websocket=self._ws, session_id=session.session_id
        )

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._heartbeat is not None

        while self._heartbeat.is_alive:
            try:
                raw = await asyncio.wait_for(self._ws.receive_bytes(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("receive_timeout", session_id=self._session.session_id)  # type: ignore[union-attr]
                continue

            self._heartbeat.on_any_message()

            try:
                frame = decode_frame(raw)
            except Exception as exc:
                logger.warning("frame_decode_error", error=str(exc))
                await self._ws.send_bytes(encode_error("DECODE_ERROR", str(exc)))
                continue

            await self._dispatch(frame)

    async def _dispatch(self, frame: ParsedFrame) -> None:
        match frame.msg_type:
            case MsgType.KEYFRAME:
                await self._handle_keyframe(frame)
            case MsgType.PING:
                await self._ws.send_bytes(encode_pong(frame.seq))
                logger.debug("ping_handled", seq=frame.seq)
            case MsgType.PONG:
                assert self._heartbeat is not None
                self._heartbeat.on_pong(frame.seq)
            case MsgType.CONTROL:
                await self._handle_control(frame)
            case _:
                logger.warning("unknown_msg_type", type=frame.msg_type)

    # ── Keyframe handling ─────────────────────────────────────────────────────

    async def _handle_keyframe(self, frame: ParsedFrame) -> None:
        assert self._throttle is not None
        assert self._buffer is not None
        assert self._session is not None

        # FPS throttle
        throttle_result = self._throttle.check()
        if not throttle_result.allowed:
            bp = BackpressurePayload(
                reason="fps_limit_exceeded",
                suggested_fps=throttle_result.suggested_fps,
                queue_depth=self._infer_q.depth,
            )
            await self._ws.send_bytes(encode_backpressure(bp, frame.seq))
            await self._store.increment(self._session.session_id, frames_dropped=1)
            return

        # Decode keypoints
        with tracer.start_as_current_span("ws.decode_keyframe"):
            try:
                if frame.is_json:
                    kf = decode_keyframe_json(frame.json())
                else:
                    kf = decode_keyframe_binary(frame.payload)
                kf.seq = frame.seq
                kf.ts_ms = float(frame.ts_ms)
            except Exception as exc:
                logger.warning("keyframe_decode_error", error=str(exc))
                await self._ws.send_bytes(encode_error("KEYFRAME_DECODE_ERROR", str(exc)))
                return

        # Build feature vector (reuse existing domain function)
        from glossa_common.schemas.cv import (
            HandLandmarks,
            HolisticFrame,
            PoseLandmark,
        )

        def _arr_to_pose(arr) -> list[PoseLandmark]:
            return [PoseLandmark(x=float(r[0]), y=float(r[1]), z=float(r[2]),
                                  visibility=float(r[3])) for r in arr]

        def _arr_to_hand(arr, handedness: str) -> HandLandmarks:
            lms = [PoseLandmark(x=float(r[0]), y=float(r[1]), z=float(r[2])) for r in arr]
            return HandLandmarks(landmarks=lms, handedness=handedness, confidence=1.0)

        hf = HolisticFrame(
            frame_index=kf.seq,
            timestamp_ms=kf.ts_ms,
            pose=_arr_to_pose(kf.pose),
            left_hand=_arr_to_hand(kf.left_hand, "Left") if kf.left_hand is not None else None,
            right_hand=_arr_to_hand(kf.right_hand, "Right") if kf.right_hand is not None else None,
        )
        feature = frame_to_feature_vector(hf)

        meta = FrameMeta(seq=kf.seq, ts_ms=kf.ts_ms, frame_index=kf.seq)

        # Push to sliding window
        window, was_dropped = await self._buffer.push(feature, meta)
        await self._store.increment(
            self._session.session_id,
            frames_received=1,
            frames_dropped=1 if was_dropped else 0,
        )

        if was_dropped:
            bp = BackpressurePayload(
                reason="buffer_overflow",
                suggested_fps=max(1.0, self._throttle.max_fps * 0.8),
                queue_depth=self._infer_q.depth,
            )
            await self._ws.send_bytes(encode_backpressure(bp, frame.seq))

        # ACK
        await self._ws.send_bytes(encode_ack(frame.seq))

        # If window ready → submit to inference
        if window is not None:
            task = InferenceTask(
                session_id=self._session.session_id,
                window=window,
                output_queue=self._output_q,
                trace_id=self._trace_id,
            )
            accepted = self._infer_q.submit(task)
            if not accepted:
                bp = BackpressurePayload(
                    reason="inference_queue_full",
                    suggested_fps=max(1.0, self._throttle.max_fps * 0.5),
                    queue_depth=self._infer_q.depth,
                )
                await self._ws.send_bytes(encode_backpressure(bp, frame.seq))
            else:
                await self._store.increment(self._session.session_id, windows_processed=1)

    # ── Control frame ─────────────────────────────────────────────────────────

    async def _handle_control(self, frame: ParsedFrame) -> None:
        assert self._session is not None
        assert self._throttle is not None

        try:
            ctrl = ControlPayload.model_validate(frame.json())
        except Exception as exc:
            await self._ws.send_bytes(encode_error("INVALID_CONTROL", str(exc)))
            return

        if ctrl.max_fps is not None:
            clamped = max(1.0, min(60.0, ctrl.max_fps))
            self._throttle.update_max_fps(clamped)
            logger.info("fps_updated", session_id=self._session.session_id, fps=clamped)

        if ctrl.window_size is not None or ctrl.stride is not None:
            new_ws = ctrl.window_size or self._session.window_size
            new_st = ctrl.stride or self._session.stride
            assert self._buffer is not None
            await self._buffer.reset()
            self._buffer = SlidingWindowBuffer(
                window_size=new_ws, stride=new_st, max_buffer=new_ws * 3
            )
            logger.info(
                "window_reconfigured",
                session_id=self._session.session_id,
                window_size=new_ws,
                stride=new_st,
            )

    # ── Result sender ─────────────────────────────────────────────────────────

    async def _result_sender(self) -> None:
        """Drain output queue and forward predictions to WebSocket."""
        while True:
            result = await self._output_q.get()
            try:
                if not result.labels:
                    continue

                for label, conf in zip(result.labels, result.confidences):
                    if label == "<blank>" or conf < 0.4:
                        continue

                    pred = PredictionPayload(
                        seq=result.end_seq,
                        gloss=label,
                        confidence=conf,
                        start_frame=result.start_seq,
                        end_frame=result.end_seq,
                        inference_ms=result.inference_ms,
                        queue_wait_ms=result.queue_wait_ms,
                    )
                    try:
                        await self._ws.send_bytes(encode_prediction(pred, seq=result.end_seq))
                        if self._session:
                            await self._store.increment(
                                self._session.session_id, predictions_emitted=1
                            )
                    except Exception:
                        logger.warning("send_prediction_failed", session_id=result.session_id)
                        return

                    logger.info(
                        "prediction_sent",
                        gloss=label,
                        confidence=round(conf, 3),
                        inference_ms=result.inference_ms,
                        queue_wait_ms=result.queue_wait_ms,
                    )
            finally:
                self._output_q.task_done()

    # ── Teardown ──────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        if self._heartbeat:
            self._heartbeat.stop()
        if self._session:
            await self._store.mark_disconnected(self._session.session_id)
            logger.info("session_teardown_complete", session_id=self._session.session_id)
