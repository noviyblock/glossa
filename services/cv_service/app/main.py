import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry import instrument_fastapi, setup_telemetry

from .config import get_settings
from .domain.pipeline import CVPipeline
from .inference.queue import InferenceQueue
from .infrastructure.mediapipe_adapter import MediaPipeHolisticAdapter
from .infrastructure.onnx_runner import build_classifier
from .routers.v1 import gesture_ws, health, recognition
from .session.redis_session import RedisSessionStore

logger = get_logger(__name__)
settings = get_settings()

_NUM_INFERENCE_WORKERS = 1   # increase for GPU multi-stream
_INFERENCE_QUEUE_SIZE = 32


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(level=settings.log_level, fmt=settings.log_format, service_name=settings.service_name)
    logger.info("starting_service", service=settings.service_name)

    setup_telemetry(
        service_name=settings.service_name,
        service_version=settings.service_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_rate=settings.otel_traces_sampler_arg,
    )

    # ── ML models ─────────────────────────────────────────────────────────────
    classifier = build_classifier(
        backend=settings.model_backend,
        model_path=settings.model_path,
        device=settings.device,
    )
    await classifier.warmup()

    mediapipe = MediaPipeHolisticAdapter(
        model_complexity=settings.mediapipe_model_complexity,
        min_detection=settings.min_detection_confidence,
        min_tracking=settings.min_tracking_confidence,
    )
    await mediapipe.ensure_initialized()

    # ── Batch inference queue (shared across all WS sessions) ─────────────────
    inference_queue = InferenceQueue(
        classifier=classifier,
        maxsize=_INFERENCE_QUEUE_SIZE,
        num_workers=_NUM_INFERENCE_WORKERS,
    )
    await inference_queue.start()

    # ── Redis + session store ──────────────────────────────────────────────────
    redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=20,
    )
    await redis_client.ping()
    session_store = RedisSessionStore(redis_client)

    # ── Wire to app state ─────────────────────────────────────────────────────
    app.state.pipeline = CVPipeline(
        classifier=classifier,
        sequence_length=settings.sequence_length,
        stride=settings.sequence_stride,
    )
    app.state.mediapipe = mediapipe
    app.state.inference_queue = inference_queue
    app.state.session_store = session_store
    app.state.redis = redis_client
    app.state.start_time = time.time()

    logger.info(
        "cv_service_ready",
        backend=settings.model_backend,
        device=settings.device,
        inference_workers=_NUM_INFERENCE_WORKERS,
        queue_size=_INFERENCE_QUEUE_SIZE,
    )
    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("cv_service_stopping")
    await inference_queue.stop()
    mediapipe.close()
    await redis_client.aclose()
    logger.info("cv_service_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glossa CV Service",
        description="MediaPipe Holistic + ONNX/OpenVINO gesture recognition with WebSocket streaming",
        version=settings.service_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    app.include_router(health.router, tags=["Health"])
    app.include_router(recognition.router, prefix="/api/v1", tags=["Recognition"])
    app.include_router(gesture_ws.router, prefix="/api/v1", tags=["Streaming"])

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    instrument_fastapi(app)
    return app


app = create_app()
