"""HTTP client for all internal Glossa microservices.

Service endpoints (configured via env):
  ASR  :8002 → POST /api/v1/transcribe
  NLP  :8003 → POST /api/v1/chat | /translate/gloss | /simplify
  TTS  :8004 → POST /api/v1/synthesize/audio
  CV   :8001 → POST /api/v1/recognize

Retry: 3 attempts, exponential backoff via tenacity.
Timeout: per-service configurable (default 30s).
"""

from __future__ import annotations

import base64
from uuid import UUID, uuid4

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from glossa_common.logging import get_logger

from ..domain.interfaces import PipelineClientPort

logger = get_logger(__name__)

_RETRY_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError)


def _retry():
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )


class GlossaPipelineClient(PipelineClientPort):
    """Fan-out HTTP client for ASR / NLP / TTS / CV services."""

    def __init__(
        self,
        asr_url: str,
        nlp_url: str,
        tts_url: str,
        cv_url: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._asr_url = asr_url.rstrip("/")
        self._nlp_url = nlp_url.rstrip("/")
        self._tts_url = tts_url.rstrip("/")
        self._cv_url  = cv_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s)
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        logger.info(
            "glossa_pipeline_client_started",
            asr=self._asr_url, nlp=self._nlp_url,
            tts=self._tts_url, cv=self._cv_url,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    # ── ASR ───────────────────────────────────────────────────────────────────

    async def transcribe(
        self, audio_bytes: bytes, session_id: str, language: str = "ru"
    ) -> str:
        """Send audio bytes to ASR service, return transcript text."""
        assert self._client is not None
        payload = {
            "session_id": session_id,
            "audio_bytes": base64.b64encode(audio_bytes).decode(),
            "language": language,
            "vad_filter": True,
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._asr_url}/api/v1/transcribe", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text", "")
                logger.debug("asr_result", chars=len(text), session=session_id)
                return text
        return ""

    # ── NLP ───────────────────────────────────────────────────────────────────

    async def chat(
        self, text: str, session_id: str, domain: str = "general", history: str = ""
    ) -> str:
        assert self._client is not None
        payload = {
            "session_id": session_id,
            "input_text": text,
            "domain": domain,
            "max_tokens": 512,
            "temperature": 0.1,
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._nlp_url}/api/v1/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("output_text", "")
        return ""

    async def translate_gloss(
        self, gloss_sequence: str, session_id: str, domain: str = "general"
    ) -> str:
        assert self._client is not None
        payload = {
            "session_id": session_id,
            "gloss_sequence": gloss_sequence,
            "domain": domain,
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._nlp_url}/api/v1/translate/gloss", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("translated_text", "")
        return ""

    async def simplify(
        self, text: str, session_id: str, domain: str = "medical"
    ) -> str:
        assert self._client is not None
        payload = {
            "session_id": session_id,
            "text": text,
            "domain": domain,
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._nlp_url}/api/v1/simplify", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("simplified_text", "")
        return ""

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def synthesize(
        self, text: str, voice_id: str = "aidar", sample_rate: int = 24000
    ) -> bytes:
        assert self._client is not None
        payload = {
            "text": text,
            "voice_id": voice_id,
            "sample_rate": sample_rate,
            "format": "wav",
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._tts_url}/api/v1/synthesize/audio", json=payload
                )
                resp.raise_for_status()
                return resp.content
        return b""

    # ── CV (Gesture) ──────────────────────────────────────────────────────────

    async def recognize_gesture(
        self, video_bytes: bytes, session_id: str
    ) -> str:
        """Send video bytes to CV service, return RSL gloss sequence."""
        assert self._client is not None
        payload = {
            "session_id": session_id,
            "video_bytes": base64.b64encode(video_bytes).decode(),
        }
        async for attempt in _retry():
            with attempt:
                resp = await self._client.post(
                    f"{self._cv_url}/api/v1/recognize", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                # CV service returns top-1 gloss or sequence
                gloss = data.get("gloss_sequence") or data.get("gloss", "")
                logger.debug("cv_result", gloss=gloss[:60], session=session_id)
                return gloss
        return ""
