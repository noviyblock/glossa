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
from gesture_segmenter import GestureSegmenter
from keypoint_extractor import KeypointExtractor, TrackState
from keypoint_smoother import KeypointSmoother
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
_session_segmenters: dict[str, GestureSegmenter] = {}
_session_smoothers: dict[str, KeypointSmoother] = {}
_session_tracks: dict[str, TrackState] = {}
_session_locks: dict[str, asyncio.Lock] = {}
_diag_counters: dict[str, int] = {}


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
    h, w = frame.shape[:2]

    # Per-session state (GestureSegmenter/KeypointSmoother/TrackState) is
    # mutated below and is NOT safe under concurrent access — a lock keeps
    # frames for the same session strictly serialized even if the client (or
    # a future pipelining change) ever has more than one request for that
    # session in flight at once. Different sessions still run fully in
    # parallel (each gets its own lock).
    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        track = _session_tracks.setdefault(session_id, TrackState())
        kp = await asyncio.to_thread(_extractor.extract, frame, track)

        if cfg.SMOOTHING_ENABLED:
            smoother = _session_smoothers.setdefault(session_id, KeypointSmoother())
            kp = smoother.smooth(kp, time.monotonic())

        person_detected = bool(np.any(kp != 0))
        kp_list = kp.tolist()  # (75, 3) – sent to client for skeleton overlay

        seg = _session_segmenters.setdefault(session_id, GestureSegmenter(session_id=session_id))

        # Periodic diagnostic log (every 30 frames) — activity score included
        # for empirical threshold tuning (see GESTURE_* constants in config.py).
        _diag_counters[session_id] = _diag_counters.get(session_id, 0) + 1
        if _diag_counters[session_id] % 30 == 0:
            nonzero = int(np.count_nonzero(kp[:, 0]))
            logger.info(
                "DIAG session=%s %s size=%dx%d person=%s nonzero_kp=%d/75",
                session_id[:8], seg.debug_state, w, h, person_detected, nonzero,
            )

        window, gesture_active, is_preview = seg.push(kp)

        if window is None:
            return {"session_id": session_id, "glosses": [], "keypoints": kp_list,
                    "person_detected": person_detected, "gesture_active": gesture_active,
                    "preview": False}

        norm_win = _normalizer(window)
        # TTA only for the FINAL classification — previews stay single-pass
        # for speed, since they're provisional anyway and get superseded.
        if cfg.TTA_ENABLED and not is_preview:
            results = await asyncio.to_thread(_classifier.predict_top3_tta, norm_win)
        else:
            results = await asyncio.to_thread(_classifier.predict_top3, norm_win)

        logger.info("session=%s top1=%s conf=%.2f gesture_active=%s preview=%s",
                    session_id[:8], results[0]["gloss"], results[0]["prob"], gesture_active, is_preview)

        return {"session_id": session_id, "glosses": results, "keypoints": kp_list,
                "person_detected": person_detected, "gesture_active": gesture_active,
                "preview": is_preview}


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
