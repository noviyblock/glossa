"""Voice message handler.

Modes:
  SPEECH_TEXT → ASR → NLP simplify → send text
  SPEECH_TTS  → ASR → NLP simplify → TTS → send audio + text caption
"""

from __future__ import annotations

import time

from glossa_common.logging import get_logger

from ..domain.entities import MAXMessage, TranslationMode, TranslationResult
from ..domain.interfaces import MAXAPIPort, PipelineClientPort
from ..session.session_manager import SessionManager

logger = get_logger(__name__)

_FALLBACK_TEXT = "Не удалось распознать голосовое сообщение. Попробуйте ещё раз."


class VoiceMessageHandler:
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
            "voice_handler_start",
            user_id=user_id,
            mode=session.mode,
            trace_id=trace_id,
        )

        # Get audio attachment URL
        from ..domain.entities import MediaType
        attachment = (
            message.body.get_attachment(MediaType.AUDIO)
            or message.body.get_attachment(MediaType.VIDEO)  # some clients send as video
        )
        if attachment is None or not attachment.url:
            await self._api.send_text(chat_id, "❌ Аудио не найдено в сообщении.")
            return TranslationResult(
                trace_id=trace_id, mode=session.mode, latency_ms=0
            )

        await self._api.send_typing(chat_id)

        # Step 1: Download audio
        try:
            audio_bytes = await self._api.download_media(attachment.url)
        except Exception:
            logger.exception("voice_download_failed", trace_id=trace_id)
            await self._api.send_text(chat_id, "❌ Не удалось скачать аудио.")
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        # Step 2: ASR transcription
        try:
            asr_text = await self._pipeline.transcribe(
                audio_bytes=audio_bytes,
                session_id=session.session_id,
                language=session.language,
            )
        except Exception:
            logger.exception("voice_asr_failed", trace_id=trace_id)
            await self._api.send_text(chat_id, _FALLBACK_TEXT)
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        if not asr_text.strip():
            await self._api.send_text(chat_id, "🔇 Речь не распознана. Говорите чётче.")
            return TranslationResult(trace_id=trace_id, mode=session.mode, latency_ms=0)

        logger.debug("voice_asr_result", chars=len(asr_text), trace_id=trace_id)

        # Step 3: NLP simplification
        try:
            simplified = await self._pipeline.simplify(
                text=asr_text,
                session_id=session.session_id,
                domain=session.domain,
            )
            if not simplified:
                simplified = asr_text
        except Exception:
            logger.exception("voice_nlp_failed", trace_id=trace_id)
            simplified = asr_text

        session.add_turn("user", f"[голос] {asr_text}")
        session.add_turn("assistant", simplified)

        audio_bytes_out: bytes | None = None

        # Step 4: Optional TTS output
        if session.mode == TranslationMode.SPEECH_TTS:
            try:
                audio_bytes_out = await self._pipeline.synthesize(
                    text=simplified,
                    voice_id="aidar",
                )
            except Exception:
                logger.exception("voice_tts_failed", trace_id=trace_id)

        await self._sessions.save(session)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("voice_handler_done", latency_ms=round(latency_ms, 1), trace_id=trace_id)

        # Send response
        asr_caption = f"🎙️ Распознано: {asr_text[:200]}\n\n✏️ Упрощено: {simplified}"
        if audio_bytes_out:
            await self._api.send_audio(
                chat_id, audio_bytes_out, caption=asr_caption, trace_id=trace_id
            )
        else:
            await self._api.send_text(chat_id, asr_caption, trace_id=trace_id)

        return TranslationResult(
            trace_id=trace_id,
            mode=session.mode,
            text_output=simplified,
            audio_bytes=audio_bytes_out,
            latency_ms=latency_ms,
            asr_text=asr_text,
            from_pipeline="asr+nlp" + ("+tts" if audio_bytes_out else ""),
        )
