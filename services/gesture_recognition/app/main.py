"""Gesture Recognition Service — FastAPI application entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry.otel import instrument_fastapi, setup_telemetry

from .config import GestureRecognitionSettings, get_settings
from .domain.sliding_window import SlidingWindowManager
from .registry.mlflow_tracker import MLflowTracker, NoopMLflowTracker
from .registry.model_loader import ModelLoader
from .registry.model_registry import ModelRegistry
from .repository.glossary_repository import GlossaryRepository
from .repository.model_repository import FilesystemModelRepository
from .routers.v1 import benchmark, health, models, recognize
from .service.batch_processor import BatchProcessor
from .service.benchmark_service import BenchmarkService
from .service.fallback_heuristics import FallbackHeuristics
from .service.recognition_service import GestureRecognitionService, RecognitionConfig

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: GestureRecognitionSettings = app.state.settings

    # Glossary
    glossary = GlossaryRepository(
        labels_path=cfg.labels_path,
        frequency_path=cfg.frequency_path,
    )
    labels = glossary.get_labels()

    # Model registry
    registry = ModelRegistry(
        labels=labels,
        feature_dim=cfg.feature_dim,
        n_threads=cfg.n_onnx_threads,
        warmup_iterations=cfg.warmup_iterations,
    )

    # MLflow tracker
    tracker = (
        MLflowTracker(cfg.mlflow_tracking_uri, cfg.mlflow_experiment)
        if cfg.mlflow_enabled
        else NoopMLflowTracker()
    )

    # Model repository + file watcher
    model_repo = FilesystemModelRepository(
        models_root=cfg.models_root,
        labels_fallback=cfg.labels_path,
        poll_interval_s=cfg.model_poll_interval_s,
    )

    # Auto-scan and load models
    specs = model_repo.scan()
    if specs:
        logger.info("autoloading_models", n=len(specs))
        await registry.load_many(specs)
    elif cfg.autoload_model_id:
        logger.warning("autoload_model_not_found", model_id=cfg.autoload_model_id)

    # Services
    try:
        active_engine = registry.get_active()
    except RuntimeError:
        active_engine = None

    fallback = FallbackHeuristics(
        labels=labels,
        confidence_floor=cfg.fallback_confidence_floor,
        top_k=cfg.top_k,
    )

    sliding_window = SlidingWindowManager(
        window_size=cfg.window_size,
        stride=cfg.window_stride,
        max_buffer=cfg.max_buffer_frames,
        feature_dim=cfg.feature_dim,
    )

    batch_proc = BatchProcessor(
        engine=active_engine,
        max_batch_size=cfg.max_batch_size,
        timeout_ms=cfg.batch_timeout_ms,
    ) if active_engine else None

    if batch_proc:
        await batch_proc.start()

    recognition_svc = GestureRecognitionService(
        registry=registry,
        batch_processor=batch_proc,
        fallback=fallback,
        sliding_window=sliding_window,
        config=RecognitionConfig(
            top_k=cfg.top_k,
            confidence_threshold=cfg.confidence_threshold,
            use_fallback=cfg.use_fallback,
        ),
    ) if batch_proc else None

    benchmark_svc = BenchmarkService(
        registry=registry,
        mlflow_tracker=tracker,
        feature_dim=cfg.feature_dim,
        sequence_length=cfg.sequence_length,
    )

    # Start file watcher
    async def _on_model_change(spec):
        logger.info("model_file_changed", model_id=spec.id)

    await model_repo.start_watching(_on_model_change)

    # Attach to app state
    app.state.registry = registry
    app.state.recognition_service = recognition_svc
    app.state.benchmark_service = benchmark_svc
    app.state.glossary = glossary
    app.state.model_repo = model_repo

    logger.info(
        "gesture_recognition_ready",
        loaded_models=registry.loaded_count,
        active_model=registry.active_model_id,
    )

    yield

    # Shutdown
    await model_repo.stop_watching()
    if batch_proc:
        await batch_proc.stop()
    logger.info("gesture_recognition_shutdown")


def create_app(settings: GestureRecognitionSettings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(level=cfg.log_level, as_json=cfg.log_json)
    setup_telemetry(service_name=cfg.service_name, otlp_endpoint=cfg.otlp_endpoint)

    app = FastAPI(
        title="Glossa Gesture Recognition Service",
        description="ONNX/OpenVINO inference for Russian Sign Language gesture recognition",
        version="1.0.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.settings = cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    instrument_fastapi(app)

    app.include_router(health.router)
    app.include_router(recognize.router)
    app.include_router(models.router)
    app.include_router(benchmark.router)

    return app


app = create_app()
