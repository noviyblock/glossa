import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import orjson
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.messaging import RedisStreamPublisher
from glossa_common.schemas.base import ErrorDetail, ErrorResponse
from glossa_common.telemetry import instrument_fastapi, setup_telemetry

from .config import get_settings
from .routers.v1 import health, translation, websocket

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        service_name=settings.service_name,
    )
    logger.info("starting_service", service=settings.service_name, env=settings.environment)

    setup_telemetry(
        service_name=settings.service_name,
        service_version=settings.service_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_rate=settings.otel_traces_sampler_arg,
    )

    app.state.redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=20,
    )
    await app.state.redis.ping()
    logger.info("redis_connected")

    app.state.publisher = RedisStreamPublisher(app.state.redis)

    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_connections // 2,
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=settings.http_timeout,
        limits=limits,
        headers={"X-Internal-Service": settings.service_name},
    )
    logger.info("service_ready", service=settings.service_name)

    yield

    logger.info("shutting_down", service=settings.service_name)
    await app.state.http_client.aclose()
    await app.state.redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glossa RSL Translation API",
        description="Multimodal Russian Sign Language translation platform",
        version=settings.service_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
        default_response_class=_ORJSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: object) -> Response:
        import uuid
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        from structlog.contextvars import bind_contextvars, clear_contextvars

        clear_contextvars()
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        response: Response = await call_next(request)  # type: ignore[operator]
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request_handled",
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 2))
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", exc_type=type(exc).__name__)
        error = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
            )
        )
        return JSONResponse(
            status_code=500,
            content=orjson.loads(error.model_dump_json()),
        )

    app.include_router(health.router, tags=["Health"])
    app.include_router(translation.router, prefix="/api/v1", tags=["Translation"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    instrument_fastapi(app)

    return app


class _ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return orjson.dumps(content)


app = create_app()
