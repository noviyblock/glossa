from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("asr_service")

_transcriber: Transcriber | None = None
_ready = False
_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber, _ready
    _transcriber = Transcriber()
    _ready = True
    yield


app = FastAPI(title="asr-service", version="0.1.0", lifespan=lifespan)

# ── Prometheus metrics ───────────────────────────────────────────────────── #

_SERVICE_NAME = "asr-service"

REQUEST_COUNT = Counter(
    "glossa_requests_total", "Total number of requests processed",
    ["service", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "glossa_request_latency_seconds", "Request latency in seconds",
    ["service", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
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


# ── Health ────────────────────────────────────────────────────────────────── #

@app.get("/health/live")
async def liveness():
    return {"service": "asr-service", "status": "healthy", "uptime": round(time.time() - _START, 2)}


@app.get("/health/ready")
async def readiness():
    if _ready:
        return {"service": "asr-service", "status": "healthy"}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"service": "asr-service", "status": "loading"},
    )


@app.get("/health")
async def health():
    return await liveness()


# ── POST /transcribe ──────────────────────────────────────────────────────── #

@app.post("/transcribe")
async def transcribe(body: dict):
    """Transcribe base64-encoded WAV/PCM audio.

    Body: {"data": "<base64>", "session_id": "..."}
    Response: {"text": str, "language": "ru", "duration_ms": int, "latency_ms": float}
    """
    raw = base64.b64decode(body["data"])
    session_id = body.get("session_id", "")

    t0 = time.perf_counter()
    result = await asyncio.to_thread(_transcriber.transcribe_bytes, raw)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "transcribe latency=%.1fms duration=%dms session=%s text=%r",
        latency_ms, result["duration_ms"], session_id, result["text"][:60],
    )
    return {**result, "latency_ms": latency_ms}


# ── WebSocket /ws ─────────────────────────────────────────────────────────── #

@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket):
    """Streaming ASR over WebSocket.

    Client → server (JSON):
        {"type": "audio_chunk", "data": "<base64 WAV/PCM>", "session_id": "..."}
        {"type": "ping"}

    Server → client (JSON):
        {"type": "partial",  "text": str, "session_id": "..."}
        {"type": "final",    "text": str, "language": "ru", "duration_ms": int, "latency_ms": float}
        {"type": "pong"}
        {"type": "error",    "message": str}
    """
    await ws.accept()
    session_id = "unknown"

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
            except TimeoutError:
                await ws.send_json({"type": "ping"})
                continue

            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg.get("type") != "audio_chunk":
                continue

            session_id = msg.get("session_id", session_id)
            raw = base64.b64decode(msg["data"])

            t0 = time.perf_counter()

            # Emit partial results as segments arrive
            def _stream(raw: bytes = raw) -> list[str]:
                parts = []
                for partial_text in _transcriber.transcribe_stream(raw):
                    parts.append(partial_text)
                return parts

            segments = await asyncio.to_thread(_stream)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            for seg in segments[:-1]:
                await ws.send_json({"type": "partial", "text": seg, "session_id": session_id})

            full_text = " ".join(segments).strip()
            duration_ms = int(len(raw) / 2 / 16000 * 1000)  # rough estimate from int16 at 16kHz
            await ws.send_json({
                "type": "final",
                "text": full_text,
                "language": "ru",
                "duration_ms": duration_ms,
                "latency_ms": latency_ms,
                "session_id": session_id,
            })
            logger.info("ws transcribe latency=%.1fms session=%s text=%r", latency_ms, session_id, full_text[:60])

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception:
        logger.exception("WS error: session=%s", session_id)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": "internal error"})
