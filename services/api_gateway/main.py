from __future__ import annotations

import contextlib
import logging
import time
import uuid

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import httpx
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
import redis.asyncio as aioredis

from call_manager import CallManager
from config import HTTP_TIMEOUT, REDIS_URL
from models import (
    AsrRequest,
    AsrResponse,
    CallCreateRequest,
    CallCreateResponse,
    CallJoinRequest,
    CallJoinResponse,
    TranslateRequest,
    TranslateResponse,
)
from orchestrator import Orchestrator
from ws_handler import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("api_gateway")

# ── Globals ────────────────────────────────────────────────────────────────── #
_orchestrator: Orchestrator  | None = None
_sessions:     SessionManager | None = None
_calls:        CallManager    | None = None
_redis:        aioredis.Redis | None = None
_http:         httpx.AsyncClient | None = None
_START = time.time()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _sessions, _calls, _redis, _http

    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _http  = httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=httpx.Limits(max_connections=100))
    _sessions = SessionManager()
    _calls = CallManager(_redis)
    _orchestrator = Orchestrator(_redis, _http)
    await _orchestrator.setup_streams()
    logger.info("API Gateway ready")
    yield
    await _http.aclose()
    await _redis.aclose()


app = FastAPI(title="api-gateway", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ───────────────────────────────────────────────────── #

_SERVICE_NAME = "api-gateway"

REQUEST_COUNT = Counter(
    "glossa_requests_total", "Total number of requests processed",
    ["service", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "glossa_request_latency_seconds", "Request latency in seconds",
    ["service", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
TRANSLATION_LATENCY = Histogram(
    "glossa_translation_latency_seconds", "End-to-end translation latency",
    ["mode", "domain"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
ACTIVE_WEBSOCKETS = Gauge(
    "glossa_active_websocket_connections", "Number of active WebSocket connections",
    ["service"],
)


@app.middleware("http")
async def _prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(service=_SERVICE_NAME, endpoint=request.url.path, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(service=_SERVICE_NAME, endpoint=request.url.path).observe(duration)
    return response


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Health ─────────────────────────────────────────────────────────────────── #

@app.get("/health/live")
async def liveness():
    return {
        "service": "api-gateway",
        "status": "healthy",
        "uptime": round(time.time() - _START, 2),
        "active_sessions": _sessions.active_count if _sessions else 0,
    }


@app.get("/health/ready")
async def readiness():
    statuses = await _orchestrator.check_services()
    all_ok = all(s == "healthy" for s in statuses.values())
    code = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=code,
        content={"service": "api-gateway", "status": "healthy" if all_ok else "degraded", "services": statuses},
    )


# ── REST /api/v1/asr — mic-input transcription only ──────────────────────────── #

@app.post("/api/v1/asr", response_model=AsrResponse)
async def asr(req: AsrRequest):
    """Speech to text only. Deliberately separate from /api/v1/translate --
    the client records a mic clip, POSTs it here to get Russian text back,
    then feeds that text through the exact same /api/v1/translate
    text_to_rsl call typed text already uses, rather than this endpoint
    re-running NLP/TTS/video/skeleton a second, parallel way."""
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        text = await _orchestrator.transcribe(session_id, req.audio)
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return AsrResponse(text=text, latency_ms=round((time.perf_counter() - t0) * 1000, 1))


# ── REST /api/v1/translate ─────────────────────────────────────────────────── #

@app.post("/api/v1/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """Synchronous translation (no streaming).

    rsl_to_text: supply gloss_sequence → returns Russian translation
    text_to_rsl: supply text (Russian) → returns gloss sequence + base64 WAV
    """
    session_id = req.session_id or str(uuid.uuid4())
    _t0 = time.perf_counter()
    try:
        result = await _orchestrator.translate_sync(
            req.mode,
            gloss_sequence=req.gloss_sequence,
            text=req.text,
            session_id=session_id,
        )
        TRANSLATION_LATENCY.labels(mode=req.mode, domain="general").observe(time.perf_counter() - _t0)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except httpx.TimeoutException as exc:
        logger.error("Upstream timeout session=%s: %s", session_id, exc)
        return JSONResponse(
            status_code=504,
            content={"error": f"Upstream service timed out: {exc}"},
        )
    except httpx.HTTPStatusError as exc:
        logger.error("Upstream error session=%s: %s", session_id, exc)
        return JSONResponse(
            status_code=502,
            content={"error": f"Upstream service error: {exc}"},
        )

    # Two-party call relay (see call_manager.py): the WS handler below has
    # its own relay for the streaming audio_chunk path, but this client's
    # actual text_to_rsl feature goes through THIS REST endpoint (typed
    # text, not a mic stream) — the WS-only relay would silently never fire
    # for it, so it's repeated here for the one mode that produces
    # something worth showing a call partner.
    if req.mode == "text_to_rsl":
        peer_id = await _calls.get_peer(session_id)
        if peer_id:
            await _sessions.send(peer_id, {
                "type": "peer_message",
                "payload": {
                    "text": result["translation"],
                    "audio": result.get("audio_wav") or None,
                    "video": result.get("video_mp4"),
                    "skeleton_sequences": result.get("skeleton_sequences"),
                },
                "session_id": peer_id,
            })

    return TranslateResponse(
        translation=result["translation"],
        glosses=result.get("glosses"),
        audio_wav=result.get("audio_wav"),
        video_mp4=result.get("video_mp4"),
        skeleton_sequences=result.get("skeleton_sequences"),
        latency_ms=result["latency_ms"],
    )


# ── REST /api/v1/call — two-party live call pairing ─────────────────────────── #
#
# MVP (see punch-list plan item 8): pairs two session_ids so the WS handler
# below can relay each side's finished translation to the other participant.
# Both participants still connect to the existing per-direction WS endpoint
# as normal (one in rsl_to_text mode, one in text_to_rsl mode) — this only
# adds the missing "who's my call partner" lookup, not a new transport.

@app.post("/api/v1/call/create", response_model=CallCreateResponse)
async def create_call(req: CallCreateRequest):
    session_id = req.session_id or str(uuid.uuid4())
    call_id = await _calls.create_call(session_id)
    return CallCreateResponse(call_id=call_id, session_id=session_id)


@app.post("/api/v1/call/{call_id}/join", response_model=CallJoinResponse)
async def join_call(call_id: str, req: CallJoinRequest):
    session_id = req.session_id or str(uuid.uuid4())
    ok = await _calls.join_call(call_id.upper(), session_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "call not found or expired"})
    return CallJoinResponse(call_id=call_id.upper(), session_id=session_id)


# ── WebSocket /api/v1/ws/translate/{mode} ─────────────────────────────────── #

@app.websocket("/api/v1/ws/translate/{mode}")
async def ws_translate(ws: WebSocket, mode: str):
    """Streaming translation WebSocket.

    mode: "rsl_to_text" | "text_to_rsl"

    Client → server:
        {"type": "video_frame", "frame": "<base64>",       "session_id": "uuid"}
        {"type": "audio_chunk", "audio": "<base64 PCM>",   "session_id": "uuid"}
        {"type": "flush_sentence",      "session_id": "uuid"}  # rsl_to_text: send buffered gestures now
        {"type": "delete_last_gesture", "session_id": "uuid"}  # rsl_to_text: drop last buffered gesture
        {"type": "end_session", "session_id": "uuid"}

    Server → client:
        {"type": "gloss",   "payload": {"glosses": [...], "confidence": 0.85,
                                         "gesture_active": bool, "preview": bool},
                                                                                "session_id": "..."}
        {"type": "pending_sentence", "payload": {"positions": [[{"gloss","prob"},...],...]},
                                                                                "session_id": "..."}
        {"type": "chunk",   "payload": {"text": str, "is_final": false},        "session_id": "..."}
        {"type": "result",  "payload": {"text": str, "confidence": float},      "session_id": "..."}
        {"type": "audio",   "payload": {"wav": "<base64>"},                     "session_id": "..."}
        {"type": "video",   "payload": {"video": "<base64 mp4>"},               "session_id": "..."}
        {"type": "skeleton", "payload": {"sequences": [{"gloss","frames"},...]},  "session_id": "..."}
        {"type": "error",   "payload": {"message": str},                        "session_id": "..."}
        {"type": "peer_message", "payload": {"text": str, "audio": str|None, "video": str|None},
                                                                                  "session_id": "..."}

    `pending_sentence` is sent whenever the buffered-gesture-sequence changes
    (a gesture was appended, or the buffer just flushed) — the client uses it
    to render "sentence so far" chips and lets the user correct/send early
    via flush_sentence/delete_last_gesture instead of only waiting for the
    SENTENCE_PAUSE_SECONDS auto-flush fallback (see orchestrator.py::process_frame).

    `peer_message` is a two-party call feature (see call_manager.py, POST
    /api/v1/call/create and /api/v1/call/{call_id}/join) — this session's
    own WS connection is unchanged either way; if it happens to be paired
    with another session_id via a call, its finished translations are also
    pushed to that other session's WS as `peer_message`, alongside (not
    instead of) the normal `result`/`audio`/`video` messages sent back to
    the session that produced them.
    """
    if mode not in ("rsl_to_text", "text_to_rsl"):
        await ws.close(code=4000, reason=f"unknown mode: {mode}")
        return

    await ws.accept()
    ACTIVE_WEBSOCKETS.labels(service=_SERVICE_NAME).inc()
    session_id: str | None = None

    async def _send(msg_type: str, payload: dict) -> None:
        await ws.send_json({"type": msg_type, "payload": payload, "session_id": session_id or ""})

    async def _relay_to_peer(payload: dict) -> None:
        """Two-party call (see call_manager.py): push a finished translation
        to the other participant's WS, if this session is paired in a call.
        No-op (not an error) if unpaired — every session works standalone
        exactly as before this feature existed."""
        peer_id = await _calls.get_peer(session_id)
        if peer_id:
            await _sessions.send(peer_id, {"type": "peer_message", "payload": payload, "session_id": peer_id})

    try:
        while True:
            msg = await ws.receive_json()
            msg_type   = msg.get("type")
            session_id = msg.get("session_id") or session_id or str(uuid.uuid4())

            # ── RSL → Text ─────────────────────────────────────────────── #
            if msg_type == "video_frame" and mode == "rsl_to_text":
                frame_b64 = msg.get("frame", "")
                if not frame_b64:
                    await _send("error", {"message": "missing frame data"})
                    continue

                if not _sessions.is_active(session_id):
                    _sessions.register(session_id, ws)

                try:
                    result = await _orchestrator.process_frame(session_id, frame_b64)
                except RuntimeError as exc:
                    await _send("error", {"message": str(exc)})
                    continue

                glosses    = result["glosses"]
                confidence = result["confidence"]

                # 1. Send gloss results immediately (+ keypoints for skeleton overlay)
                await _send("gloss", {
                    "glosses": glosses,
                    "confidence": confidence,
                    "keypoints": result.get("keypoints"),
                    "person_detected": result.get("person_detected", False),
                    "gesture_active": result.get("gesture_active", False),
                    "preview": result.get("preview", False),
                    "gesture_keypoints": result.get("gesture_keypoints"),
                })

                # 2. Send text translation as partial + final, then voice it
                # (a hearing person watching a signer otherwise only ever
                # sees text, never hears the translation).
                translation = result["translation"]
                if translation:
                    await _send("chunk", {"text": translation, "is_final": False})
                    await _send("result", {"text": translation, "confidence": confidence})
                    wav_b64 = result.get("wav_b64")
                    if wav_b64:
                        await _send("audio", {"wav": wav_b64})
                    await _relay_to_peer({"text": translation, "audio": wav_b64 or None})

                # 3. Buffered-sentence state, only when it actually changed
                # this frame (gesture appended, or buffer just auto-flushed)
                if "pending_positions" in result:
                    await _send("pending_sentence", {"positions": result["pending_positions"]})

            # ── RSL → Text: manual sentence controls ──────────────────────── #
            elif msg_type == "flush_sentence" and mode == "rsl_to_text":
                flush_result = await _orchestrator.flush_pending_sentence(session_id)
                translation = flush_result["translation"]
                if translation:
                    await _send("chunk", {"text": translation, "is_final": False})
                    await _send("result", {"text": translation, "confidence": 1.0})
                    wav_b64 = flush_result.get("wav_b64")
                    if wav_b64:
                        await _send("audio", {"wav": wav_b64})
                    await _relay_to_peer({"text": translation, "audio": wav_b64 or None})
                await _send("pending_sentence", {"positions": flush_result["pending_positions"]})

            elif msg_type == "delete_last_gesture" and mode == "rsl_to_text":
                positions = await _orchestrator.delete_last_pending(session_id)
                await _send("pending_sentence", {"positions": positions})

            # ── Text → RSL ─────────────────────────────────────────────── #
            elif msg_type == "audio_chunk" and mode == "text_to_rsl":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    await _send("error", {"message": "missing audio data"})
                    continue

                if not _sessions.is_active(session_id):
                    _sessions.register(session_id, ws)

                try:
                    result = await _orchestrator.process_audio(session_id, audio_b64)
                except RuntimeError as exc:
                    await _send("error", {"message": str(exc)})
                    continue

                russian_text   = result["text"]
                gloss_sequence = result["gloss_sequence"]
                wav_b64        = result["wav_b64"]
                video_b64      = result.get("video_b64")
                skeleton_seqs  = result.get("skeleton_sequences")

                # 1. Recognised text (partial chunk)
                if russian_text:
                    await _send("chunk", {"text": russian_text, "is_final": False})

                # 2. RSL gloss sequence (final result)
                await _send("result", {"text": gloss_sequence, "confidence": 1.0})

                # 3. TTS audio output
                if wav_b64:
                    await _send("audio", {"wav": wav_b64})

                # 4. Sign-clip video output — best-effort, None if none of the
                # glosses matched a reference clip (see SignVideoAssembler).
                if video_b64:
                    await _send("video", {"video": video_b64})

                # 4b. Skeleton keypoint sequences — best-effort alternative
                # to the video, for clients that play back a skeleton
                # instead (see SkeletonSequenceProvider).
                if skeleton_seqs:
                    await _send("skeleton", {"sequences": skeleton_seqs})

                # 5. Relay to call partner, if any -- one message carrying
                # everything (gloss text + audio + video/skeleton) so the
                # peer's client handles it as one incoming turn, not several.
                if gloss_sequence or wav_b64 or video_b64 or skeleton_seqs:
                    await _relay_to_peer({
                        "text": gloss_sequence, "audio": wav_b64 or None, "video": video_b64,
                        "skeleton_sequences": skeleton_seqs,
                    })

            # ── End session ─────────────────────────────────────────────── #
            elif msg_type == "end_session":
                if session_id:
                    _sessions.unregister(session_id)
                    await _orchestrator.delete_session(session_id)
                break

            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "session_id": session_id or ""})

            else:
                await _send("error", {"message": f"unexpected message type '{msg_type}' for mode '{mode}'"})

    except WebSocketDisconnect:
        logger.info("WS disconnected session=%s", session_id)
    except Exception:
        logger.exception("WS error session=%s", session_id)
        with contextlib.suppress(Exception):
            await _send("error", {"message": "internal server error"})
    finally:
        ACTIVE_WEBSOCKETS.labels(service=_SERVICE_NAME).dec()
        if session_id:
            _sessions.unregister(session_id)
