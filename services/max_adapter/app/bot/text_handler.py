"""Text message handler — mode: TEXT_RESPONSE → NLP chat."""

from __future__ import annotations

import time

from glossa_common.logging import get_logger

from ..domain.entities import MAXMessage, TranslationMode, TranslationResult
from ..domain.interfaces import MAXAPIPort, PipelineClientPort
from ..session.session_manager import SessionManager

logger = get_logger(__name__)

_FALLBACK = "Извините, не удалось обработать запрос. Попробуйте позже."


class TextMessageHandler:
    def __init__(
        self,
        pipeline: PipelineClientPort,
        session_manager: SessionManager,
        max_api: MAXAPIPort,
    ) -> None:
        self._pipeline = pipeline
        self._sessions = session_manager
        self._api = max_api

    async def handle(self, message: MAXMessage) -> TranslationResult:
        user_id = message.sender.user_id
        chat_id = message.recipient_chat_id
        trace_id = message.trace_id
        t0 = time.perf_counter()

        session = await self._sessions.get_or_create(user_id, chat_id)
        input_text = message.body.text.strip()

        logger.info(
            "text_handler_start",
            user_id=user_id,
            mode=session.mode,
            chars=len(input_text),
            trace_id=trace_id,
        )

        session.add_turn("user", input_text)

        try:
            await self._api.send_typing(chat_id)
            response = await self._pipeline.chat(
                text=input_text,
                session_id=session.session_id,
                domain=session.domain,
                history=session.history_text(),
            )
            if not response:
                response = _FALLBACK
        except Exception:
            logger.exception("text_handler_pipeline_error", trace_id=trace_id)
            response = _FALLBACK

        session.add_turn("assistant", response)
        await self._sessions.save(session)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "text_handler_done",
            user_id=user_id,
            latency_ms=round(latency_ms, 1),
            trace_id=trace_id,
        )

        await self._api.send_text(chat_id, response, trace_id=trace_id)
        return TranslationResult(
            trace_id=trace_id,
            mode=TranslationMode.TEXT_RESPONSE,
            text_output=response,
            latency_ms=latency_ms,
            from_pipeline="nlp",
        )
