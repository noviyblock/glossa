"""Webhook endpoint — receives MAX platform updates."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request, Response, status

from glossa_common.logging import get_logger

from ...domain.entities import UpdateType
from ...infrastructure.webhook_security import verify_webhook_signature
from ...queue.event_queue import QueuedEvent

logger = get_logger(__name__)

router = APIRouter()


async def _rate_check(rate_limiter, user_id: int) -> tuple[bool, int]:
    """Unified async/sync consume call across limiter backends."""
    import asyncio
    consume = rate_limiter.consume
    if asyncio.iscoroutinefunction(consume):
        return await consume(user_id)
    return consume(user_id)



@router.post("/webhook", status_code=status.HTTP_200_OK)
async def webhook(request: Request) -> Response:
    """Accept MAX platform update, validate signature, enqueue for processing."""
    state = request.app.state
    secret: str | None = getattr(state, "webhook_secret", None)

    # Signature verification (reads body)
    try:
        body_bytes = await verify_webhook_signature(request, secret)
    except Exception as exc:
        logger.warning("webhook_signature_failed", error=str(exc))
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    import json as _json
    try:
        payload: dict = _json.loads(body_bytes)
    except Exception:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    update_type_str: str = payload.get("update_type", "")
    trace_id = str(uuid4())[:8]

    logger.debug("webhook_received", update_type=update_type_str, trace_id=trace_id)

    # Rate limiting (best-effort — user_id from message)
    try:
        update_type = UpdateType(update_type_str)
    except ValueError:
        logger.debug("webhook_unknown_update_type", update_type=update_type_str)
        return Response(status_code=status.HTTP_200_OK)

    if update_type == UpdateType.MESSAGE_CREATED:
        msg_data = payload.get("message", {})
        user_id = msg_data.get("from", msg_data.get("sender", {})).get("user_id")
        if user_id:
            rate_limiter = getattr(state, "rate_limiter", None)
            if rate_limiter:
                allowed, _ = await _rate_check(rate_limiter, user_id)
                if not allowed:
                    logger.info("webhook_rate_limited", user_id=user_id, trace_id=trace_id)
                    # Return 200 to MAX but silently drop
                    return Response(status_code=status.HTTP_200_OK)

    event = QueuedEvent(
        update_type=update_type_str,
        payload=payload,
        trace_id=trace_id,
    )
    event_queue = getattr(state, "event_queue", None)
    if event_queue:
        enqueued = await event_queue.enqueue(event)
        if not enqueued:
            logger.warning("webhook_queue_full", trace_id=trace_id)

    return Response(status_code=status.HTTP_200_OK)
