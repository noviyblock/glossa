"""Webhook signature verification for MAX Bot API.

MAX signs webhook payloads with HMAC-SHA256 using the bot token as the secret.
Expected header: X-MAX-Signature: sha256=<hex_digest>

If the secret is not configured, verification is skipped (dev mode).
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request, status

from glossa_common.logging import get_logger

logger = get_logger(__name__)

_SIGNATURE_HEADER = "X-MAX-Signature"
_SIGNATURE_PREFIX = "sha256="


def _compute_signature(body: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode(), body, hashlib.sha256)
    return _SIGNATURE_PREFIX + mac.hexdigest()


async def verify_webhook_signature(
    request: Request,
    secret: str | None,
) -> bytes:
    """Read body, verify HMAC-SHA256 signature, return raw bytes.

    Raises HTTP 401 on signature mismatch.
    Skips verification if secret is None or empty (dev/test mode).
    """
    body = await request.body()

    if not secret:
        logger.debug("webhook_signature_check_skipped_no_secret")
        return body

    header_value = request.headers.get(_SIGNATURE_HEADER, "")
    if not header_value.startswith(_SIGNATURE_PREFIX):
        logger.warning(
            "webhook_missing_signature",
            header=_SIGNATURE_HEADER,
            received=header_value[:32],
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed webhook signature",
        )

    expected = _compute_signature(body, secret)
    received = header_value

    if not hmac.compare_digest(expected, received):
        logger.warning("webhook_signature_mismatch")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Webhook signature verification failed",
        )

    return body
