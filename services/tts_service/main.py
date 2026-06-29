from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from synthesizer import Synthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("tts_service")

_synthesizer: Synthesizer | None = None
_ready = False
_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _synthesizer, _ready
    _synthesizer = Synthesizer()
    _ready = True
    yield


app = FastAPI(title="tts-service", version="0.1.0", lifespan=lifespan)

# ── Prometheus metrics ───────────────────────────────────────────────────── #

_SERVICE_NAME = "tts-service"

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
    return {"service": "tts-service", "status": "healthy", "uptime": round(time.time() - _START, 2)}


@app.get("/health/ready")
async def readiness():
    if _ready:
        return {
            "service": "tts-service",
            "status": "healthy",
            "cache_size": len(_synthesizer._cache),
        }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"service": "tts-service", "status": "loading"},
    )


@app.get("/health")
async def health():
    return await liveness()


# ── GET /voices ───────────────────────────────────────────────────────────── #

@app.get("/voices")
async def voices():
    """Return list of available TTS speaker voices."""
    return {"voices": _synthesizer.voices}


# ── POST /synthesize ──────────────────────────────────────────────────────── #

@app.post("/synthesize")
async def synthesize(body: dict):
    """Synthesize text to speech.

    Body:     {"text": str, "speaker": "xenia"}
    Response: {"audio": "<base64 WAV>", "cached": bool, "latency_ms": float}
    """
    text    = body.get("text", "").strip()
    speaker = body.get("speaker", "xenia")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is empty"})

    hits_before = _synthesizer._cache.hits

    t0 = time.perf_counter()
    wav_bytes = await asyncio.to_thread(_synthesizer.synthesize, text, speaker)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    was_cached = _synthesizer._cache.hits > hits_before
    audio_b64 = base64.b64encode(wav_bytes).decode()

    logger.info(
        "synthesize latency=%.1fms cached=%s speaker=%s text=%r",
        latency_ms, was_cached, speaker, text[:60],
    )
    return {"audio": audio_b64, "cached": was_cached, "latency_ms": latency_ms}
