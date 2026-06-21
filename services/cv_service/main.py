from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

import config as cfg
from gesture_classifier import GestureClassifier
from keypoint_extractor import KeypointExtractor
from normalizer import Normalizer
from sliding_window import SlidingWindowBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("cv_service")

# ── Globals (initialised in lifespan) ────────────────────────────────────── #
_extractor:  KeypointExtractor  | None = None
_normalizer: Normalizer         | None = None
_classifier: GestureClassifier  | None = None
_redis:      aioredis.Redis     | None = None
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _extractor, _normalizer, _classifier, _redis, _ready
    logger.info("Loading models…")
    _extractor  = KeypointExtractor()
    _normalizer = Normalizer(cfg.NORM_STATS_PATH)
    _classifier = GestureClassifier(
        ov_xml_path=cfg.OV_XML_PATH,
        onnx_path=cfg.ONNX_MOBILE_PATH,
        class_map_path=cfg.CLASS_MAP_PATH,
    )
    _redis = aioredis.from_url(cfg.REDIS_URL, decode_responses=True)
    _ready = True
    logger.info("CV service ready")
    yield
    if _extractor:
        _extractor.close()
    if _redis:
        await _redis.aclose()


app = FastAPI(title="cv-service", version="0.1.0", lifespan=lifespan)

# ── Health endpoints ──────────────────────────────────────────────────────── #

_START = time.time()


@app.get("/health/live")
async def liveness():
    return {"service": "cv-service", "status": "healthy", "uptime": round(time.time() - _START, 2)}


@app.get("/health/ready")
async def readiness():
    if _ready:
        return {"service": "cv-service", "status": "healthy"}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"service": "cv-service", "status": "loading"},
    )


# ── REST endpoint (single frame) ─────────────────────────────────────────── #

@app.post("/process_frame")
async def process_frame(body: dict[str, Any]):
    """Accept a single base64 JPEG frame and return top-3 glosses.

    Body: {"session_id": str, "data": "<base64-jpeg>"}
    """
    import cv2

    raw = base64.b64decode(body["data"])
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "cannot decode image"})

    kp = await asyncio.to_thread(_extractor.extract, frame)

    # Build a dummy full window by repeating the single frame
    window = np.stack([kp] * cfg.WINDOW_SIZE, axis=0)
    window = _normalizer(window)
    results = await asyncio.to_thread(_classifier.predict_top3, window)

    return {"session_id": body.get("session_id", ""), "glosses": results}


# ── WebSocket endpoint ────────────────────────────────────────────────────── #

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Streaming WebSocket.

    Client → server: JSON frames
        {"type": "frame", "data": "<base64-jpeg>", "session_id": "..."}
        {"type": "ping"}

    Server → client: JSON results
        {"type": "result", "session_id": "...", "glosses": [...], "timestamp": ...}
        {"type": "pong"}
    """
    import cv2

    await ws.accept()
    session_id = None
    buf = SlidingWindowBuffer(
        window_size=cfg.WINDOW_SIZE,
        stride=cfg.WINDOW_STRIDE,
        conf_threshold=cfg.CONF_THRESHOLD,
    )

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

            if msg.get("type") != "frame":
                continue

            session_id = msg.get("session_id", session_id or "unknown")
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            kp = await asyncio.to_thread(_extractor.extract, frame)
            window = buf.push(kp)
            if window is None:
                continue

            norm_win = await asyncio.to_thread(_normalizer, window)
            glosses  = await asyncio.to_thread(_classifier.predict_top3, norm_win)

            top1 = glosses[0]
            buf.on_result(top1["gloss"], top1["prob"])

            ts = time.time()
            payload = {"session_id": session_id, "glosses": glosses, "timestamp": ts}

            await ws.send_json({"type": "result", **payload})

            if _redis:
                await _redis.xadd(
                    cfg.CV_STREAM_OUT,
                    {"payload": json.dumps(payload)},
                    maxlen=1000,
                    approximate=True,
                )

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception:
        logger.exception("WS error: session=%s", session_id)
        await ws.close(code=1011)
