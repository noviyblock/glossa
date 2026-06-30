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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

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
_session_buffers: dict[str, SlidingWindowBuffer] = {}


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

# ── Prometheus metrics ───────────────────────────────────────────────────── #

_SERVICE_NAME = "cv-service"

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

    session_id = body.get("session_id", "")
    kp = await asyncio.to_thread(_extractor.extract, frame)

    person_detected = bool(np.any(kp != 0))

    buf = _session_buffers.setdefault(session_id, SlidingWindowBuffer())
    buf_len_before = len(buf._buf)
    window = buf.push(kp)

    if window is None:
        # Log every 10 frames to avoid log flood
        if buf_len_before % 10 == 0:
            logger.info(
                "session=%s person=%s kp_nonzero=%d/%d buf=%d/%d (waiting for full window)",
                session_id[:8], person_detected,
                int(np.count_nonzero(kp[:, :2])), len(kp),
                len(buf._buf), cfg.WINDOW_SIZE,
            )
        return {"session_id": session_id, "glosses": []}

    norm_win = _normalizer(window)
    results = await asyncio.to_thread(_classifier.predict_top3, norm_win)
    buf.on_result(results[0]["gloss"], results[0]["prob"])

    logger.info(
        "session=%s person=%s → top1=%s conf=%.3f top2=%s conf=%.3f",
        session_id[:8], person_detected,
        results[0]["gloss"], results[0]["prob"],
        results[1]["gloss"] if len(results) > 1 else "-",
        results[1]["prob"]  if len(results) > 1 else 0,
    )

    return {"session_id": session_id, "glosses": results}


@app.post("/debug/frame")
async def debug_frame(body: dict[str, Any]):
    """Diagnostic endpoint: returns full pipeline state for a single frame.

    Body: {"data": "<base64-jpeg>", "session_id": str (optional)}
    Returns detailed info without requiring a full window.
    """
    import cv2

    raw = base64.b64decode(body.get("data", ""))
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if len(arr) else None

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "cannot decode image"})

    session_id = body.get("session_id", "debug")
    kp = await asyncio.to_thread(_extractor.extract, frame)

    nonzero_joints = int(np.count_nonzero(kp[:, :2].any(axis=1)))
    person_detected = nonzero_joints > 0

    buf = _session_buffers.get(session_id)
    buf_size = len(buf._buf) if buf else 0

    # Always run inference on available data (pad if needed) for diagnostics
    if buf_size > 0 and buf is not None:
        available = np.stack(list(buf._buf), axis=0)
    else:
        available = kp[np.newaxis]  # single frame as fallback

    # Pad/trim to WINDOW_SIZE for inference
    T = cfg.WINDOW_SIZE
    if len(available) >= T:
        w = available[-T:]
    else:
        repeats = (T // len(available)) + 1
        w = np.tile(available, (repeats, 1, 1))[:T]

    norm_win = _normalizer(w)
    results = await asyncio.to_thread(_classifier.predict_top3, norm_win)

    return {
        "config": {
            "window_size": cfg.WINDOW_SIZE,
            "window_stride": cfg.WINDOW_STRIDE,
            "conf_threshold": cfg.CONF_THRESHOLD,
            "det_conf": cfg.DWPOSE_DET_CONF,
        },
        "frame": {
            "person_detected": person_detected,
            "nonzero_joints": nonzero_joints,
            "total_joints": len(kp),
            "kp_mean_xy": float(np.abs(kp[:, :2]).mean()),
        },
        "buffer": {
            "current_size": buf_size,
            "needed": T,
            "filled_pct": round(buf_size / T * 100, 1),
        },
        "prediction_now": results,
        "session_id": session_id,
    }


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
