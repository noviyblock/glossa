from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from skeleton import SkeletonSequenceProvider
from synthesizer import Synthesizer
from video import SignVideoAssembler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("tts_service")

_synthesizer: Synthesizer | None = None
_video: SignVideoAssembler | None = None
_skeleton: SkeletonSequenceProvider | None = None
_ready = False
_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _synthesizer, _video, _skeleton, _ready
    _synthesizer = Synthesizer()
    _video = SignVideoAssembler()
    try:
        _skeleton = SkeletonSequenceProvider()
    except FileNotFoundError as exc:
        # processed_64_200 is a DVC-tracked dataset directory, not a small
        # config file -- missing on a host that hasn't run `dvc pull` for
        # it. Same non-fatal-degradation pattern as SignVideoAssembler's
        # own "no clip matched" case: /skeleton_sequence just returns an
        # empty list instead of taking the whole service down.
        logger.warning("SkeletonSequenceProvider unavailable (%s) — /skeleton_sequence will return no frames", exc)
        _skeleton = None
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


# ── POST /sign_video ──────────────────────────────────────────────────────── #

@app.post("/sign_video")
async def sign_video(body: dict):
    """Render a gloss sequence as a video of concatenated reference sign clips.

    Body:     {"gloss_sequence": "ПРИВЕТ КАК ДЕЛА"}
    Response: {"video": "<base64 MP4>" | null, "total": int, "latency_ms": float}

    `video` is null if none of the glosses in the sequence had a matching
    clip on disk (see SignVideoAssembler — best-effort normalized matching,
    not a guaranteed hit).
    """
    gloss_sequence = body.get("gloss_sequence", "").strip()
    if not gloss_sequence:
        return JSONResponse(status_code=400, content={"error": "gloss_sequence is empty"})

    total = len(gloss_sequence.split())

    t0 = time.perf_counter()
    video_bytes = await asyncio.to_thread(_video.build, gloss_sequence)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    video_b64 = base64.b64encode(video_bytes).decode() if video_bytes else None
    logger.info(
        "sign_video latency=%.1fms matched=%s gloss_sequence=%r",
        latency_ms, video_b64 is not None, gloss_sequence[:60],
    )
    return {"video": video_b64, "total": total, "latency_ms": latency_ms}


# ── POST /skeleton_sequence ───────────────────────────────────────────────── #

@app.post("/skeleton_sequence")
async def skeleton_sequence(body: dict):
    """Render a gloss sequence as per-gloss skeleton keypoint sequences for
    client-side playback (see SkeletonSequenceProvider) — an alternative to
    /sign_video's concatenated clip, reusing the same per-gesture keypoint
    data the ST-GCN classifier trains on.

    Body:     {"gloss_sequence": "ПРИВЕТ КАК ДЕЛА"}
    Response: {"sequences": [{"gloss": str, "frames": [[[x,y,conf],...75],...T]}, ...],
               "total": int, "latency_ms": float}

    `sequences` omits tokens with no matching sample — same best-effort
    semantics as /sign_video, not a guaranteed hit for every gloss.
    """
    gloss_sequence = body.get("gloss_sequence", "").strip()
    if not gloss_sequence:
        return JSONResponse(status_code=400, content={"error": "gloss_sequence is empty"})

    total = len(gloss_sequence.split())

    t0 = time.perf_counter()
    sequences = (
        await asyncio.to_thread(_skeleton.get, gloss_sequence) if _skeleton is not None else []
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "skeleton_sequence latency=%.1fms matched=%d/%d gloss_sequence=%r",
        latency_ms, len(sequences), total, gloss_sequence[:60],
    )
    return {"sequences": sequences, "total": total, "latency_ms": latency_ms}
