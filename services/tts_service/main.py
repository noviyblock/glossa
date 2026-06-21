from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

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
