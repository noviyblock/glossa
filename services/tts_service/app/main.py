"""FastAPI application entry point — wires all TTS components."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry import instrument_fastapi, setup_telemetry

from .audio.encoder import AudioEncoder
from .cache.phrase_cache import InMemoryPhraseCache, RedisPhraseCache, TieredPhraseCache
from .config import get_settings
from .domain.entities import AudioFormat
from .domain.text_processor import TextPreprocessor
from .pipeline.tts_pipeline import TTSPipeline
from .queue.synthesis_queue import SynthesisQueue
from .routers.v1 import benchmark, cache, health, synthesize, ws
from .synthesis.silero_runner import SileroTTSRunner

logger = get_logger(__name__)
settings = get_settings()


def _build_cache(cfg) -> TieredPhraseCache:
    l1 = InMemoryPhraseCache(
        max_entries=cfg.cache_max_entries,
        max_bytes=cfg.cache_max_bytes,
    )
    l2 = RedisPhraseCache(
        redis_url=cfg.cache_redis_url,
        ttl_seconds=cfg.cache_ttl_seconds,
    )
    return TieredPhraseCache(l1=l1, l2=l2)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        service_name=settings.service_name,
    )
    logger.info("starting_tts_service", service=settings.service_name)

    setup_telemetry(
        service_name=settings.service_name,
        service_version=settings.service_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_rate=settings.otel_traces_sampler_arg,
    )

    # Synthesis engine
    runner = SileroTTSRunner(
        model_dir=settings.model_dir,
        language=settings.language,
        device=settings.device,
        cpu_threads=settings.cpu_threads,
        warmup_on_load=settings.warmup_on_load,
        warmup_voice_id=settings.default_voice_id,
    )
    await runner.ensure_loaded()
    logger.info("silero_ready", device=settings.device, voice=settings.default_voice_id)

    # Tiered cache
    phrase_cache = _build_cache(settings)
    await phrase_cache.start()
    logger.info("phrase_cache_ready", backend=settings.cache_backend)

    # Audio encoder (stateless)
    encoder = AudioEncoder()

    # Synthesis queue
    queue = SynthesisQueue(
        synthesizer=runner,
        workers=settings.queue_workers,
        maxsize=settings.queue_maxsize,
        job_timeout_s=settings.queue_job_timeout_s,
    )
    await queue.start()
    logger.info("synthesis_queue_ready", workers=settings.queue_workers)

    # High-level pipeline
    pipeline = TTSPipeline(
        synthesizer=runner,
        cache=phrase_cache,
        encoder=encoder,
        queue=queue,
        preprocessor=TextPreprocessor(),
    )

    # Preload common phrases into cache
    if settings.preload_phrases_on_start:
        loaded = await pipeline.preload_phrases(
            voice_id=settings.default_voice_id,
            fmt=AudioFormat(settings.default_format),
            sample_rate=settings.default_sample_rate,
        )
        logger.info("phrases_preloaded", count=loaded)

    app.state.pipeline = pipeline
    app.state.runner = runner
    app.state.start_time = time.time()

    logger.info("tts_service_ready")
    yield

    logger.info("tts_service_stopping")
    await queue.stop()
    await phrase_cache.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glossa TTS Service",
        description="Silero TTS — streaming speech synthesis for the RSL pipeline",
        version=settings.service_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(synthesize.router, prefix="/api/v1", tags=["Synthesis"])
    app.include_router(cache.router, prefix="/api/v1", tags=["Cache"])
    app.include_router(benchmark.router, prefix="/api/v1", tags=["Benchmark"])
    app.include_router(ws.router, prefix="/api/v1", tags=["WebSocket"])

    app.mount("/metrics", make_asgi_app())
    instrument_fastapi(app)

    return app


app = create_app()
