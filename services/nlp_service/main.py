from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from cache import LRUCache
from translator import Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("nlp_service")

# ── Globals (initialised in lifespan) ────────────────────────────────────── #
_translator: Translator | None = None
_cache:      LRUCache   | None = None
_ready = False

_START = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _translator, _cache, _ready
    logger.info("Loading Qwen2-1.5B…")
    _translator = Translator()
    _cache      = LRUCache()
    _ready      = True
    logger.info("NLP service ready")
    yield


app = FastAPI(title="nlp-service", version="0.1.0", lifespan=lifespan)

# ── Prometheus metrics ───────────────────────────────────────────────────── #

_SERVICE_NAME = "nlp-service"

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
    return {"service": "nlp-service", "status": "healthy", "uptime": round(time.time() - _START, 2)}


@app.get("/health/ready")
async def readiness():
    if _ready:
        return {"service": "nlp-service", "status": "healthy"}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"service": "nlp-service", "status": "loading"},
    )


# ── POST /translate ───────────────────────────────────────────────────────── #

@app.post("/translate")
async def translate(body: dict[str, Any]):
    """Translate a gloss sequence to Russian.

    Body: {"gloss_sequence": "ПРИВЕТ КАК ДЕЛА", "domain": "general", "session_id": "..."}
    """
    gloss_sequence = body.get("gloss_sequence", "").strip().upper()
    session_id     = body.get("session_id", "")

    cached = _cache.get(gloss_sequence)
    if cached is not None:
        logger.info("translate cache_hit session=%s glosses=%r", session_id, gloss_sequence)
        return {"translation": cached, "cached": True, "latency_ms": 0.0}

    t0 = time.perf_counter()
    translation = await asyncio.to_thread(_translator.translate, gloss_sequence)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

    _cache.set(gloss_sequence, translation)
    logger.info(
        "translate latency=%.1fms session=%s glosses=%r",
        latency_ms, session_id, gloss_sequence,
    )
    return {"translation": translation, "cached": False, "latency_ms": latency_ms}


# ── POST /translate_topk ──────────────────────────────────────────────────── #

@app.post("/translate_topk")
async def translate_topk(body: dict[str, Any]):
    """Translate using top-k hypotheses with probabilities.

    Body: {"hypotheses": [{"gloss": "ПРИВЕТ", "prob": 0.85}, ...], "domain": "general",
           "context": ["previous translated sentence", ...]}

    `context` (optional) — recent translated sentences from this session, most
    recent last. Lets the LLM disambiguate among ambiguous gloss candidates by
    semantic coherence with the ongoing conversation, not just raw confidence.
    Skips the cache when context is present, since the same hypotheses can now
    legitimately translate differently depending on what came before.
    """
    hypotheses = body.get("hypotheses", [])
    context    = body.get("context") or []
    if not hypotheses:
        return JSONResponse(status_code=400, content={"error": "hypotheses list is empty"})

    cache_key = "|".join(f"{h['gloss']}:{h['prob']:.3f}" for h in hypotheses)
    if not context:
        cached = _cache.get(cache_key)
        if cached is not None:
            return {"translation": cached, "cached": True, "latency_ms": 0.0}

    t0 = time.perf_counter()
    translation = await asyncio.to_thread(_translator.translate_topk, hypotheses, context)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

    if not context:
        _cache.set(cache_key, translation)
    logger.info("translate_topk latency=%.1fms top1=%r context_turns=%d",
                latency_ms, hypotheses[0].get("gloss"), len(context))
    return {"translation": translation, "cached": False, "latency_ms": latency_ms}


# ── POST /translate_sequence_topk ─────────────────────────────────────────── #

@app.post("/translate_sequence_topk")
async def translate_sequence_topk(body: dict[str, Any]):
    """Translate a whole buffered gesture sequence (sentence) in one call.

    Body: {"positions": [[{"gloss": "Я", "prob": 0.9}, ...], ...],
           "context": ["previous translated sentence", ...]}

    Each element of `positions` is the top-k candidate list for one
    recognised gesture, in temporal order — generalizes /translate_topk
    (single gesture) to a full sentence: the LLM picks the most
    linguistically coherent combination across positions instead of the
    best candidate for each gesture in isolation. Always skips the cache —
    a given set of per-position candidates is rarely repeated verbatim.
    """
    positions = body.get("positions", [])
    context   = body.get("context") or []
    if not positions:
        return JSONResponse(status_code=400, content={"error": "positions list is empty"})

    t0 = time.perf_counter()
    translation = await asyncio.to_thread(_translator.translate_sequence_topk, positions, context)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

    logger.info("translate_sequence_topk latency=%.1fms n_positions=%d context_turns=%d",
                latency_ms, len(positions), len(context))
    return {"translation": translation, "cached": False, "latency_ms": latency_ms}


# ── POST /translate_reverse ───────────────────────────────────────────────── #

@app.post("/translate_reverse")
async def translate_reverse(body: dict[str, Any]):
    """Reverse translation: Russian sentence → RSL gloss sequence.

    Body: {"text": "Привет, как дела?", "session_id": "..."}
    Response: {"translation": "ПРИВЕТ КАК ДЕЛА", "cached": bool, "latency_ms": float}
    """
    text       = body.get("text", "").strip()
    session_id = body.get("session_id", "")
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is empty"})

    cache_key = f"rev|{text}"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("translate_reverse cache_hit session=%s", session_id)
        return {"translation": cached, "cached": True, "latency_ms": 0.0}

    t0 = time.perf_counter()
    translation = await asyncio.to_thread(_translator.translate_reverse, text)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

    _cache.set(cache_key, translation)
    logger.info("translate_reverse latency=%.1fms session=%s text=%r", latency_ms, session_id, text[:60])
    return {"translation": translation, "cached": False, "latency_ms": latency_ms}


# ── GET /health (alias) ───────────────────────────────────────────────────── #

@app.get("/health")
async def health_alias():
    return await liveness()
