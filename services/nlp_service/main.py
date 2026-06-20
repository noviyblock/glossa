from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

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

    Body: {"hypotheses": [{"gloss": "ПРИВЕТ", "prob": 0.85}, ...], "domain": "general"}
    """
    hypotheses = body.get("hypotheses", [])
    if not hypotheses:
        return JSONResponse(status_code=400, content={"error": "hypotheses list is empty"})

    # Cache key: canonical string of sorted gloss+prob pairs
    cache_key = "|".join(f"{h['gloss']}:{h['prob']:.3f}" for h in hypotheses)
    cached = _cache.get(cache_key)
    if cached is not None:
        return {"translation": cached, "cached": True, "latency_ms": 0.0}

    t0 = time.perf_counter()
    translation = await asyncio.to_thread(_translator.translate_topk, hypotheses)
    latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

    _cache.set(cache_key, translation)
    logger.info("translate_topk latency=%.1fms top1=%r", latency_ms, hypotheses[0].get("gloss"))
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
