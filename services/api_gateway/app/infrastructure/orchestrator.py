import time
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import orjson

from glossa_common.logging import get_logger
from glossa_common.schemas.asr import ASRResult
from glossa_common.schemas.cv import GestureRecognitionResult
from glossa_common.schemas.nlp import NLPResult
from glossa_common.schemas.translation import TranslationChunk, TranslationMode, TranslationRequest, TranslationResult
from glossa_common.schemas.tts import TTSResult

logger = get_logger(__name__)


class TranslationOrchestrator:
    def __init__(self, client: httpx.AsyncClient, settings: object) -> None:
        self._client = client
        self._settings = settings

    async def run(self, request: TranslationRequest) -> TranslationResult:
        t_start = time.perf_counter()
        stage_latencies: dict[str, float] = {}

        if request.mode == TranslationMode.RSL_TO_TEXT:
            result = await self._rsl_to_text(request, stage_latencies)
        else:
            result = await self._text_to_rsl(request, stage_latencies)

        total_ms = (time.perf_counter() - t_start) * 1000
        result.total_latency_ms = round(total_ms, 2)
        result.stage_latencies = stage_latencies
        return result

    async def _rsl_to_text(
        self, request: TranslationRequest, latencies: dict[str, float]
    ) -> TranslationResult:
        from ...config import get_settings  # avoid circular at module level
        cfg = get_settings()

        # CV: gesture recognition
        t0 = time.perf_counter()
        cv_resp = await self._client.post(
            f"{cfg.cv_service_url}/api/v1/recognize",
            json={"session_id": str(request.session_id), "frames": []},
        )
        cv_resp.raise_for_status()
        cv_result = GestureRecognitionResult.model_validate(cv_resp.json())
        latencies["cv"] = round((time.perf_counter() - t0) * 1000, 2)

        # NLP: translate gloss → natural language
        t0 = time.perf_counter()
        nlp_resp = await self._client.post(
            f"{cfg.nlp_service_url}/api/v1/translate",
            json={
                "session_id": str(request.session_id),
                "gloss_sequence": cv_result.raw_text,
                "domain": request.domain,
                "context_hint": request.context_hint,
            },
        )
        nlp_resp.raise_for_status()
        nlp_result = NLPResult.model_validate(nlp_resp.json())
        latencies["nlp"] = round((time.perf_counter() - t0) * 1000, 2)

        # TTS: synthesize output
        t0 = time.perf_counter()
        tts_resp = await self._client.post(
            f"{cfg.tts_service_url}/api/v1/synthesize",
            json={"session_id": str(request.session_id), "text": nlp_result.text},
        )
        tts_resp.raise_for_status()
        tts_result = TTSResult.model_validate_json(tts_resp.content)
        latencies["tts"] = round((time.perf_counter() - t0) * 1000, 2)

        return TranslationResult(
            session_id=request.session_id,
            mode=request.mode,
            text=nlp_result.text,
            audio_bytes=tts_result.audio_bytes,
            confidence=cv_result.confidence,
            total_latency_ms=0.0,
            tokens_used=nlp_result.tokens_generated,
        )

    async def _text_to_rsl(
        self, request: TranslationRequest, latencies: dict[str, float]
    ) -> TranslationResult:
        from ...config import get_settings
        cfg = get_settings()

        t0 = time.perf_counter()
        nlp_resp = await self._client.post(
            f"{cfg.nlp_service_url}/api/v1/translate",
            json={
                "session_id": str(request.session_id),
                "text": request.context_hint or "",
                "domain": request.domain,
            },
        )
        nlp_resp.raise_for_status()
        nlp_result = NLPResult.model_validate(nlp_resp.json())
        latencies["nlp"] = round((time.perf_counter() - t0) * 1000, 2)

        return TranslationResult(
            session_id=request.session_id,
            mode=request.mode,
            text=nlp_result.gloss_sequence or "",
            confidence=1.0,
            total_latency_ms=0.0,
            tokens_used=nlp_result.tokens_generated,
        )

    async def stream(
        self, request: TranslationRequest, audio_bytes: bytes
    ) -> AsyncIterator[TranslationChunk]:
        from ...config import get_settings
        cfg = get_settings()

        # ASR streaming
        chunk_index = 0
        async with self._client.stream(
            "POST",
            f"{cfg.asr_service_url}/api/v1/transcribe/stream",
            content=audio_bytes,
            headers={"Content-Type": "audio/pcm", "X-Session-ID": str(request.session_id)},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = orjson.loads(line)
                yield TranslationChunk(
                    session_id=request.session_id,
                    chunk_index=chunk_index,
                    text=data.get("text"),
                    confidence=data.get("confidence", 0.9),
                    is_final=data.get("is_final", False),
                    stage="asr",
                )
                chunk_index += 1

    async def stream_video(
        self, request: TranslationRequest, frame_data: object
    ) -> AsyncIterator[TranslationChunk]:
        # Placeholder for video frame streaming — delegates to CV service SSE
        from ...config import get_settings
        cfg = get_settings()
        yield TranslationChunk(
            session_id=request.session_id,
            chunk_index=0,
            text=None,
            confidence=0.0,
            is_final=False,
            stage="cv",
        )

    async def get_session(self, session_id: UUID) -> TranslationResult | None:
        return None
