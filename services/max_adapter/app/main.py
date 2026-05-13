"""MAX Adapter service entry-point."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry import instrument_fastapi, setup_telemetry

from .bot.event_router import EventRouter
from .config import MAXAdapterSettings, get_settings
from .domain.entities import MAXMessage, UpdateType
from .infrastructure.max_api_client import MAXAPIClient
from .infrastructure.rate_limiter import InMemoryRateLimiter, RedisRateLimiter
from .pipeline.glossa_client import GlossaPipelineClient
from .queue.event_queue import EventQueue, QueuedEvent
from .routers.v1 import health, pipeline
from .routers.v1 import webhook as webhook_router
from .routers.v1 import ws_polling
from .session.session_manager import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionManager,
    TieredSessionStore,
)

logger = get_logger(__name__)


def _build_session_manager(cfg: MAXAdapterSettings) -> SessionManager:
    if cfg.session_backend == "redis":
        store = RedisSessionStore(redis_url=cfg.redis_url, ttl_s=cfg.session_ttl_s)
    elif cfg.session_backend == "tiered":
        mem = InMemorySessionStore(max_sessions=cfg.max_memory_sessions)
        red = RedisSessionStore(redis_url=cfg.redis_url, ttl_s=cfg.session_ttl_s)
        store = TieredSessionStore(memory=mem, redis=red)
    else:
        store = InMemorySessionStore(max_sessions=cfg.max_memory_sessions)
    return SessionManager(store)


def _build_rate_limiter(cfg: MAXAdapterSettings):
    if cfg.rate_limit_backend == "redis":
        return RedisRateLimiter(
            redis_url=cfg.redis_url,
            limit=cfg.rate_limit_requests,
            window_s=cfg.rate_limit_window_s,
        )
    return InMemoryRateLimiter(
        limit=cfg.rate_limit_requests,
        window_s=cfg.rate_limit_window_s,
    )


def _parse_update(payload: dict, trace_id: str):
    """Parse raw MAX webhook payload into (UpdateType, MAXMessage | None)."""
    from .domain.entities import (
        MAXAttachment, MAXMessageBody, MAXUser, MediaType,
    )

    update_type_str = payload.get("update_type", "")
    try:
        update_type = UpdateType(update_type_str)
    except ValueError:
        return None, None

    if update_type != UpdateType.MESSAGE_CREATED:
        return update_type, None

    msg_data = payload.get("message", {})
    sender_data = msg_data.get("from", msg_data.get("sender", {}))
    body_data = msg_data.get("body", {})

    attachments = []
    for a in body_data.get("attachments", []):
        try:
            media_type = MediaType(a.get("type", "file"))
        except ValueError:
            media_type = MediaType.FILE
        attachments.append(MAXAttachment(type=media_type, payload=a.get("payload", {})))

    body = MAXMessageBody(
        mid=body_data.get("mid", ""),
        text=body_data.get("text") or "",
        attachments=attachments,
    )
    recipient = msg_data.get("recipient", {})
    message = MAXMessage(
        sender=MAXUser(
            user_id=sender_data.get("user_id", 0),
            first_name=sender_data.get("first_name") or sender_data.get("name", ""),
            last_name=sender_data.get("last_name"),
            username=sender_data.get("username"),
            is_bot=sender_data.get("is_bot", False),
        ),
        recipient_chat_id=recipient.get("chat_id") or recipient.get("user_id") or 0,
        recipient_user_id=recipient.get("user_id"),
        body=body,
        trace_id=trace_id,
    )
    return update_type, message


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = get_settings()
    configure_logging(
        level=cfg.log_level, fmt=cfg.log_format, service_name=cfg.service_name
    )
    logger.info("starting_service", service=cfg.service_name)

    setup_telemetry(
        service_name=cfg.service_name,
        service_version=cfg.service_version,
        otlp_endpoint=cfg.otel_exporter_otlp_endpoint,
        sample_rate=cfg.otel_traces_sampler_arg,
    )

    # Infrastructure
    max_api = MAXAPIClient(base_url=cfg.max_api_base_url, token=cfg.bot_token)
    await max_api.start()

    pipeline_client = GlossaPipelineClient(
        asr_url=cfg.asr_url,
        nlp_url=cfg.nlp_url,
        tts_url=cfg.tts_url,
        cv_url=cfg.cv_url,
        timeout_s=cfg.pipeline_timeout,
    )
    await pipeline_client.start()

    session_manager = _build_session_manager(cfg)
    rate_limiter = _build_rate_limiter(cfg)
    if hasattr(rate_limiter, "start"):
        await rate_limiter.start()

    event_router = EventRouter(
        pipeline=pipeline_client,
        session_manager=session_manager,
        max_api=max_api,
    )

    async def _process_event(event: QueuedEvent) -> None:
        update_type, message = _parse_update(event.payload, event.trace_id)
        if update_type is None or message is None:
            return
        try:
            await event_router.route(update_type, message)
        except Exception:
            logger.exception("event_processing_error", trace_id=event.trace_id)

    event_queue = EventQueue(
        handler=_process_event,
        workers=cfg.event_queue_workers,
        maxsize=cfg.event_queue_maxsize,
    )
    await event_queue.start()

    # Expose on app.state for routers
    app.state.max_api = max_api
    app.state.pipeline = pipeline_client
    app.state.session_manager = session_manager
    app.state.rate_limiter = rate_limiter
    app.state.event_queue = event_queue
    app.state.event_router = event_router
    app.state.webhook_secret = cfg.webhook_secret or None
    app.state.start_time = time.time()
    # Legacy compat for health endpoint
    app.state.client = None

    if cfg.webhook_url:
        try:
            await max_api.subscribe_webhook(cfg.webhook_url)
            logger.info("webhook_registered", url=cfg.webhook_url)
        except Exception:
            logger.exception("webhook_registration_failed", url=cfg.webhook_url)

    logger.info(
        "max_adapter_ready",
        session_backend=cfg.session_backend,
        rate_limit_backend=cfg.rate_limit_backend,
        queue_workers=cfg.event_queue_workers,
    )

    yield

    logger.info("max_adapter_stopping")
    await event_queue.stop()
    await pipeline_client.stop()
    await max_api.stop()
    if hasattr(rate_limiter, "stop"):
        await rate_limiter.stop()
    logger.info("max_adapter_stopped")


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="Glossa MAX Adapter",
        description="MAX Messenger bot integration for the Glossa RSL translation platform",
        version=cfg.service_version,
        docs_url="/docs" if not cfg.is_production else None,
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
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
    app.include_router(webhook_router.router, prefix="/api/v1", tags=["Webhook"])
    app.include_router(ws_polling.router, prefix="/api/v1", tags=["WebSocket"])
    app.mount("/metrics", make_asgi_app())
    instrument_fastapi(app)
    return app


app = create_app()
