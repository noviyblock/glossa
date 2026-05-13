"""Video circle handler.

Modes:
  GESTURE_TEXT  → CV recognize → NLP translate_gloss → send text
  GESTURE_VOICE → CV recognize → NLP translate_gloss → TTS → send audio + text caption
"""

from __future__ import annotations

import time

from glossa_common.logging import get_logger

from ..domain.entities import MAXMessage, MediaType, TranslationMode, TranslationResult
from ..domain.interfaces import MAXAPIPort, PipelineClientPort
from ..session.session_manager import SessionManager

logger = get_logger(__name__)

_FALLBACK_NO_GESTURE = "Жест не распознан. Попробуйте снова — держите руку в кадре."


class VideoMessageHandler:
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
        logger.info(
            "video_handler_start",
            user_id=user_id,
            mode=session.mode,
            trace_id=trace_id,
        )

        # Get video attachment (video_circle preferred, fallback to video)
        attachment = (
            message.body.get_attachment(MediaType.VIDEO_CIRCLE)
            or message.body.get_attachment(MediaType.VIDEO)
        )
        if attachment is None or not attachment.url:
            await self._api.send_text(chat_id, "❌ Видео не найдено в сообщении.")
            return TranslationResult(
                trace_id=trace_id, mode=session.mode, latency_ms=0
            )

        await self._api.send_typing(chat_id)

        # Step 1: Download video
        try:
            video_bytes = await self._api.download_media(attachment.url)
        except Exception:
            logger.exception("video_download_failed", trace_id=trace_id)
            await self._api.send_text(chat_id, "❌ Не удалось скачать видео.")
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        # Step 2: CV gesture recognition → gloss sequence
        try:
            gloss_sequence = await self._pipeline.recognize_gesture(
                video_bytes=video_bytes,
                session_id=session.session_id,
            )
        except Exception:
            logger.exception("video_cv_failed", trace_id=trace_id)
            await self._api.send_text(chat_id, _FALLBACK_NO_GESTURE)
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        if not gloss_sequence.strip():
            await self._api.send_text(chat_id, "🤷 " + _FALLBACK_NO_GESTURE)
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        logger.debug(
            "video_cv_result",
            gloss=gloss_sequence[:60],
            trace_id=trace_id,
        )

        # Step 3: NLP gloss → Russian text
        try:
            translated = await self._pipeline.translate_gloss(
                gloss_sequence=gloss_sequence,
                session_id=session.session_id,
                domain=session.domain,
            )
            if not translated:
                translated = gloss_sequence
        except Exception:
            logger.exception("video_nlp_failed", trace_id=trace_id)
            translated = gloss_sequence

        session.add_turn("user", f"[жест] {gloss_sequence}")
        session.add_turn("assistant", translated)

        audio_bytes_out: bytes | None = None

        # Step 4: Optional TTS output
        if session.mode == TranslationMode.GESTURE_VOICE:
            try:
                audio_bytes_out = await self._pipeline.synthesize(
                    text=translated,
                    voice_id="aidar",
                )
            except Exception:
                logger.exception("video_tts_failed", trace_id=trace_id)

        await self._sessions.save(session)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("video_handler_done", latency_ms=round(latency_ms, 1), trace_id=trace_id)

        # Send response
        caption = f"🤟 Глосс: {gloss_sequence[:150]}\n\n🇷🇺 Перевод: {translated}"
        if audio_bytes_out:
            await self._api.send_audio(
                chat_id, audio_bytes_out, caption=caption, trace_id=trace_id
            )
        else:
            await self._api.send_text(chat_id, caption, trace_id=trace_id)

        return TranslationResult(
            trace_id=trace_id,
            mode=session.mode,
            text_output=translated,
            audio_bytes=audio_bytes_out,
            latency_ms=latency_ms,
            gloss_sequence=gloss_sequence,
            from_pipeline="cv+nlp" + ("+tts" if audio_bytes_out else ""),
        )
