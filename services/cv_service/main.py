from __future__ import annotations

import asyncio
import base64
import contextlib
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
# monotonic timestamp of the last /process_frame call per session — the
# sweep in _cleanup_idle_sessions() below is the only reader/writer besides
# process_frame itself.
_session_last_access: dict[str, float] = {}
_cleanup_task: asyncio.Task | None = None


async def _cleanup_idle_sessions() -> None:
    """Background sweep: evict per-session state (GestureSegmenter/
    KeypointSmoother/TrackState/lock/diag counter) for sessions idle longer
    than SESSION_IDLE_TTL. Without this, these dicts grow for as long as the
    process runs — every session_id that ever connected stays resident.

    Skips a session if its lock is currently held (mid-request) rather than
    deleting out from under it — picked up on the next sweep instead.
    """
    while True:
        await asyncio.sleep(cfg.SESSION_CLEANUP_INTERVAL)
        now = time.monotonic()
        stale = [
            sid for sid, last_seen in list(_session_last_access.items())
            if now - last_seen > cfg.SESSION_IDLE_TTL
        ]
        if not stale:
            continue
        evicted = 0
        for sid in stale:
            lock = _session_locks.get(sid)
            if lock is not None and lock.locked():
                continue
            _session_segmenters.pop(sid, None)
            _session_smoothers.pop(sid, None)
            _session_tracks.pop(sid, None)
            _session_locks.pop(sid, None)
            _diag_counters.pop(sid, None)
            _session_last_access.pop(sid, None)
            evicted += 1
        if evicted:
            logger.info("Session cleanup: evicted %d idle session(s) (TTL=%ds)",
                        evicted, cfg.SESSION_IDLE_TTL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _extractor, _normalizer, _classifier, _redis, _ready, _cleanup_task
    logger.info("Loading models…")
    # onnxruntime silently falling back to CPU when RTMLIB_DEVICE=cuda was
    # requested is a documented past failure mode (see Dockerfile's deps-gpu
    # stage comments: CUDAExecutionProvider missing entirely from the
    # available-providers list, not just failing to initialize) -- log the
    # actual provider list at startup so that regression is visible instead
    # of silently degrading to CPU-speed inference under a "cuda" label.
    import onnxruntime as _ort
    _providers = _ort.get_available_providers()
    logger.info("onnxruntime available providers: %s", _providers)
    if cfg.RTMLIB_DEVICE == "cuda" and "CUDAExecutionProvider" not in _providers:
        logger.warning(
            "RTMLIB_DEVICE=cuda requested but CUDAExecutionProvider is NOT in "
            "onnxruntime's available providers -- extraction will silently run "
            "on CPU. Check nvidia-container-toolkit / --gpus / GPU build target."
        )
    _extractor  = KeypointExtractor()
    _normalizer = Normalizer(cfg.NORM_STATS_PATH)
    _classifier = GestureClassifier(
        ov_xml_path=cfg.OV_XML_PATH,
        onnx_path=cfg.ONNX_MOBILE_PATH,
        class_map_path=cfg.CLASS_MAP_PATH,
    )
    _redis = aioredis.from_url(cfg.REDIS_URL, decode_responses=True)
    _ready = True
    _cleanup_task = asyncio.create_task(_cleanup_idle_sessions())
    logger.info("CV service ready")
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
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

    # Touch before doing any work, not after — a session mid-request (however
    # long extraction/classification takes) must never look idle to the
    # cleanup sweep, and this also "revives" a session that _cleanup_idle_sessions
    # is about to (or just did) evict.
    _session_last_access[session_id] = time.monotonic()

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

        # Periodic diagnostic log (every DIAG_LOG_INTERVAL frames) — activity
        # score included for empirical threshold tuning (see GESTURE_*
        # constants in config.py).
        _diag_counters[session_id] = _diag_counters.get(session_id, 0) + 1
        if _diag_counters[session_id] % cfg.DIAG_LOG_INTERVAL == 0:
            # Per-region breakdown, not just a combined total — the combined
            # number alone can't tell you whether it's the whole person
            # dropping out (bad framing/lighting/distance) or specifically
            # the hands during motion (see HAND_LOW_CONF_ZERO_THRESHOLD).
            # scripts/analyze_cv_logs.py parses this exact "kp=body:.. lhand:..
            # rhand:.." format — keep them in sync if this line changes.
            body_nz  = int(np.count_nonzero(kp[0:33, 0]))
            lhand_nz = int(np.count_nonzero(kp[33:54, 0]))
            rhand_nz = int(np.count_nonzero(kp[54:75, 0]))
            logger.info(
                "DIAG session=%s %s size=%dx%d person=%s "
                "kp=body:%d/33 lhand:%d/21 rhand:%d/21 total:%d/75",
                session_id[:8], seg.debug_state, w, h, person_detected,
                body_nz, lhand_nz, rhand_nz, body_nz + lhand_nz + rhand_nz,
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

        response: dict = {"session_id": session_id, "glosses": results, "keypoints": kp_list,
                           "person_detected": person_detected, "gesture_active": gesture_active,
                           "preview": is_preview}
        if not is_preview:
            # The just-completed gesture's own frames (raw, pre-normalizer —
            # same [0,1]-ish coordinate convention as `keypoints` above, not
            # z-scored) — lets the client show/scrub the actual gesture that
            # was classified instead of only ever the single current live
            # frame, which (especially right after the person stops moving,
            # e.g. at the end of an uploaded video) stopped being
            # representative of the recognized sign within about a second.
            response["gesture_keypoints"] = window.tolist()
        return response


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
