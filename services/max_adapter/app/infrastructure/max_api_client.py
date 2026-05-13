"""MAX Messenger Bot API HTTP client.

Base URL: https://platform-api.max.ru
Auth:     Authorization: {token}  (header)

Covered endpoints:
  GET  /me                      → bot info
  GET  /updates                 → long-polling
  POST /messages                → send text / attachment message
  GET  /uploads?type={type}     → get pre-signed upload URL
  POST <upload_url>             → upload binary to pre-signed URL

MAX API attachment upload flow:
  1. GET /uploads?type=audio  → {"url": "...", "token": "..."}
  2. PUT/POST <url> with raw bytes
  3. Include token in message attachments

Retry strategy: 3 attempts, exponential backoff 0.5s → 1s → 2s.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from glossa_common.logging import get_logger

from ..domain.interfaces import MAXAPIPort

logger = get_logger(__name__)

_BASE_URL    = "https://platform-api.max.ru"
_TIMEOUT_S   = 20.0
_MAX_TEXT_LEN = 4000   # MAX messenger text limit per message


class MAXAPIClient(MAXAPIPort):
    """Async httpx client for the MAX Bot API."""

    def __init__(
        self,
        token: str,
        base_url: str = _BASE_URL,
        timeout_s: float = _TIMEOUT_S,
        max_retries: int = 3,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s)
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": self._token},
            timeout=self._timeout,
        )
        logger.info("max_api_client_started", base_url=self._base_url)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get_bot_info(self) -> dict:
        return await self._get("/me")

    async def send_text(self, chat_id: int, text: str, trace_id: str = "") -> None:
        """Send plain text message, splitting if over 4000-char limit."""
        chunks = self._split_text(text)
        for i, chunk in enumerate(chunks):
            # Target is a query param; body carries only content
            await self._retry_post(
                "/messages", {"text": chunk},
                params={"chat_id": chat_id},
                trace_id=trace_id,
            )
            if i < len(chunks) - 1:
                await asyncio.sleep(0.3)

    async def send_audio(
        self, chat_id: int, audio_bytes: bytes, caption: str = "", trace_id: str = ""
    ) -> None:
        """Upload audio bytes and send as voice message."""
        try:
            token = await self._upload_audio(audio_bytes)
        except Exception:
            logger.exception("max_audio_upload_failed", trace_id=trace_id)
            if caption:
                await self.send_text(chat_id, caption, trace_id=trace_id)
            return

        body: dict[str, Any] = {
            "attachments": [{"type": "audio", "payload": {"token": token}}],
        }
        if caption:
            body["text"] = caption[:_MAX_TEXT_LEN]
        await self._retry_post(
            "/messages", body,
            params={"chat_id": chat_id},
            trace_id=trace_id,
        )

    async def send_typing(self, chat_id: int) -> None:
        """Send typing indicator (fire-and-forget, no retry)."""
        try:
            assert self._client is not None
            await self._client.post(
                f"/chats/{chat_id}/actions",
                json={"action": "typing_on"},
            )
        except Exception:
            pass

    async def download_media(self, url: str) -> bytes:
        """Download media file from MAX CDN. Uses fresh client (no auth needed for CDN)."""
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
            ):
                with attempt:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp.content
        raise RuntimeError(f"Failed to download media from {url}")

    async def get_updates(self, marker: int | None = None, timeout_s: int = 30) -> dict:
        """Long-polling update fetch."""
        params: dict[str, Any] = {"timeout": timeout_s}
        if marker is not None:
            params["marker"] = marker
        return await self._get("/updates", params=params)

    async def subscribe_webhook(self, url: str) -> dict:
        """Register this bot's webhook URL with the MAX platform."""
        return await self._retry_post("/subscriptions", {"url": url})

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _upload_audio(self, audio_bytes: bytes) -> str:
        """Upload audio and return the attachment token.

        MAX audio upload flow (differs from image/file):
          1. POST /uploads?type=audio  → {url, token}  (token is pre-issued)
          2. POST <url> multipart       → server processes file
          3. Use token from step 1 in the message body
        """
        assert self._client is not None
        # Step 1: obtain pre-signed upload URL + pre-issued attachment token
        resp1 = await self._client.post("/uploads", params={"type": "audio"})
        resp1.raise_for_status()
        upload_info = resp1.json()
        upload_url: str = upload_info["url"]
        token: str = upload_info["token"]

        # Step 2: multipart upload to pre-signed URL (no auth header)
        async with httpx.AsyncClient(timeout=60.0) as cdn:
            resp2 = await cdn.post(
                upload_url,
                files={"data": ("audio.wav", audio_bytes, "audio/wav")},
            )
            resp2.raise_for_status()

        return token

    async def _get(self, path: str, params: dict | None = None) -> dict:
        assert self._client is not None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        ):
            with attempt:
                resp = await self._client.get(path, params=params)
                resp.raise_for_status()
                return resp.json()
        return {}

    async def _retry_post(
        self, path: str, payload: dict,
        params: dict | None = None,
        trace_id: str = "",
    ) -> dict:
        assert self._client is not None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        ):
            with attempt:
                resp = await self._client.post(path, json=payload, params=params)
                resp.raise_for_status()
                logger.debug(
                    "max_api_sent",
                    path=path,
                    status=resp.status_code,
                    trace_id=trace_id,
                )
                return resp.json()
        return {}

    @staticmethod
    def _split_text(text: str, limit: int = _MAX_TEXT_LEN) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:limit])
            text = text[limit:]
        return chunks
