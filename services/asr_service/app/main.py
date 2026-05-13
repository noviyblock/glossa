"""ASR Service — FastAPI application entry point."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry.otel import instrument_fastapi, setup_telemetry

from .benchmark.benchmark_service import ASRBenchmarkService
from .config import ASRSettings, get_settings
from .queue.audio_queue import AudioChunkQueue
from .routers.v1 import health
from .routers.v1.asr_ws import router as ws_router
from .routers.v1.transcribe import router as transcribe_router
from .schemas.request import BenchmarkRequest
from .schemas.response import BenchmarkResponse
from .session.asr_session import InMemorySessionStore, RedisSessionStore
from .transcription.medical_adapter import MedicalAdapter, NoopMedicalAdapter
from .transcription.whisper_runner import FasterWhisperRunner
from .vad.silero_vad import SileroVAD, create_vad

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: ASRSettings = app.state.settings
    configure_logging(level=cfg.log_level, as_json=cfg.log_json)
    setup_telemetry(service_name=cfg.service_name, otlp_endpoint=cfg.otlp_endpoint)

    # Transcriber
    transcriber = FasterWhisperRunner(
        model_size=cfg.model_size,
        device=cfg.device,
        compute_type=cfg.compute_type,
        model_dir=cfg.model_dir,
        num_workers=cfg.num_workers,
        cpu_threads=cfg.cpu_threads,
        language=cfg.language,
        beam_size=cfg.beam_size,
        best_of=cfg.best_of,
        vad_filter=True,
        vad_min_silence_ms=cfg.vad_min_silence_ms,
        word_timestamps=cfg.word_timestamps,
        initial_prompt=cfg.initial_prompt,
        condition_on_previous_text=cfg.condition_on_previous_text,
        temperature=cfg.temperature,
    )
    await transcriber.ensure_loaded()
    await transcriber.warmup(cfg.warmup_iterations)

    # VAD
    vad = create_vad(
        backend=cfg.vad_backend,
        sample_rate=16000,
        threshold=cfg.vad_threshold,
        min_speech_ms=cfg.vad_min_speech_ms,
        min_silence_ms=cfg.vad_min_silence_ms,
        model_path=cfg.vad_model_path,
    )
    if isinstance(vad, SileroVAD):
        await vad.ensure_loaded()

    # Medical adapter
    medical = (
        MedicalAdapter(languages=cfg.medical_languages)
        if cfg.medical_adaptation
        else NoopMedicalAdapter()
    )

    # Session store
    if cfg.session_backend == "redis":
        session_store = RedisSessionStore(cfg.redis_url)
    else:
        session_store = InMemorySessionStore()
    await session_store.start()

    # Audio queue
    audio_queue = AudioChunkQueue(
        maxsize=cfg.audio_queue_maxsize,
        num_workers=cfg.audio_queue_workers,
    )
    await audio_queue.start()

    # Benchmark service
    benchmark_svc = ASRBenchmarkService(transcriber=transcriber)

    # Attach to app state
    app.state.transcriber    = transcriber
    app.state.vad            = vad
    app.state.medical_adapter = medical
    app.state.session_store  = session_store
    app.state.audio_queue    = audio_queue
    app.state.benchmark_service = benchmark_svc
    app.state.start_time     = time.time()

    logger.info(
        "asr_service_ready",
        model=cfg.model_size,
        device=transcriber.device,
        compute=transcriber.compute_type,
        vad=cfg.vad_backend,
    )

    yield

    await audio_queue.stop()
    await session_store.stop()
    logger.info("asr_service_stopped")


def create_app(settings: ASRSettings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    app = FastAPI(
        title="Glossa ASR Service",
        description="Streaming speech recognition based on faster-whisper + CTranslate2",
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
    app.include_router(transcribe_router)
    app.include_router(ws_router)
    app.mount("/metrics", make_asgi_app())

    # Inline benchmark endpoint with proper DI
    from fastapi import APIRouter
    bench_router = APIRouter(prefix="/api/v1", tags=["benchmark"])

    @bench_router.post("/benchmark", response_model=BenchmarkResponse)
    async def run_benchmark(body: BenchmarkRequest, request: Request) -> BenchmarkResponse:
        svc: ASRBenchmarkService = request.app.state.benchmark_service
        result = await svc.run(
            audio_duration_s=body.audio_duration_s,
            n_samples=body.n_samples,
            language=body.language,
            beam_size=body.beam_size,
            warmup_iters=body.warmup_iters,
        )
        return BenchmarkResponse(**result.to_dict())

    app.include_router(bench_router)
    return app


app = create_app()
