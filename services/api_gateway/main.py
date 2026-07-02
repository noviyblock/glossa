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

from config import HTTP_TIMEOUT, REDIS_URL
from models import TranslateRequest, TranslateResponse
from orchestrator import Orchestrator
from ws_handler import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("api_gateway")

# ── Globals ────────────────────────────────────────────────────────────────── #
_orchestrator: Orchestrator  | None = None
_sessions:     SessionManager | None = None
_redis:        aioredis.Redis | None = None
_http:         httpx.AsyncClient | None = None
_START = time.time()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _sessions, _redis, _http

    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _http  = httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=httpx.Limits(max_connections=100))
    _sessions = SessionManager()
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

    return TranslateResponse(
        translation=result["translation"],
        glosses=result.get("glosses"),
        audio_wav=result.get("audio_wav"),
        latency_ms=result["latency_ms"],
    )


# ── WebSocket /api/v1/ws/translate/{mode} ─────────────────────────────────── #

@app.websocket("/api/v1/ws/translate/{mode}")
async def ws_translate(ws: WebSocket, mode: str):
    """Streaming translation WebSocket.

    mode: "rsl_to_text" | "text_to_rsl"

    Client → server:
        {"type": "video_frame", "frame": "<base64>",       "session_id": "uuid"}
        {"type": "audio_chunk", "audio": "<base64 PCM>",   "session_id": "uuid"}
        {"type": "end_session", "session_id": "uuid"}

    Server → client:
        {"type": "gloss",   "payload": {"glosses": [...], "confidence": 0.85,
                                         "gesture_active": bool, "preview": bool},
                                                                                "session_id": "..."}
        {"type": "chunk",   "payload": {"text": str, "is_final": false},        "session_id": "..."}
        {"type": "result",  "payload": {"text": str, "confidence": float},      "session_id": "..."}
        {"type": "audio",   "payload": {"wav": "<base64>"},                     "session_id": "..."}
        {"type": "error",   "payload": {"message": str},                        "session_id": "..."}
    """
    if mode not in ("rsl_to_text", "text_to_rsl"):
        await ws.close(code=4000, reason=f"unknown mode: {mode}")
        return

    await ws.accept()
    ACTIVE_WEBSOCKETS.labels(service=_SERVICE_NAME).inc()
    session_id: str | None = None

    async def _send(msg_type: str, payload: dict) -> None:
        await ws.send_json({"type": msg_type, "payload": payload, "session_id": session_id or ""})

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
                })

                # 2. Send text translation as partial + final
                translation = result["translation"]
                if translation:
                    await _send("chunk", {"text": translation, "is_final": False})
                    await _send("result", {"text": translation, "confidence": confidence})

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

                # 1. Recognised text (partial chunk)
                if russian_text:
                    await _send("chunk", {"text": russian_text, "is_final": False})

                # 2. RSL gloss sequence (final result)
                await _send("result", {"text": gloss_sequence, "confidence": 1.0})

                # 3. TTS audio output
                if wav_b64:
                    await _send("audio", {"wav": wav_b64})

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
