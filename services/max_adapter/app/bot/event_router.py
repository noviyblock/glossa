"""Event router — dispatches incoming MAX updates to appropriate handlers."""

from __future__ import annotations

from glossa_common.logging import get_logger

from ..domain.entities import MAXMessage, MediaType, TranslationMode, TranslationResult, UpdateType
from ..domain.interfaces import MAXAPIPort, PipelineClientPort
from ..session.session_manager import SessionManager
from .command_handler import CommandHandler
from .text_handler import TextMessageHandler
from .video_handler import VideoMessageHandler
from .voice_handler import VoiceMessageHandler

logger = get_logger(__name__)

_UNSUPPORTED_MEDIA = "⚠️ Этот тип вложения пока не поддерживается. Отправьте голосовое, видео-кружок или текст."


class EventRouter:
    def __init__(
        self,
        pipeline: PipelineClientPort,
        session_manager: SessionManager,
        max_api: MAXAPIPort,
    ) -> None:
        self._sessions = session_manager
        self._api = max_api

        self._cmd_handler = CommandHandler(session_manager, max_api)
        self._text_handler = TextMessageHandler(pipeline, session_manager, max_api)
        self._voice_handler = VoiceMessageHandler(pipeline, session_manager, max_api)
        self._video_handler = VideoMessageHandler(pipeline, session_manager, max_api)

    async def route(self, update_type: UpdateType, message: MAXMessage) -> TranslationResult | None:
        if update_type != UpdateType.MESSAGE_CREATED:
            logger.debug("event_router_skip", update_type=update_type, trace_id=message.trace_id)
            return None

        trace_id = message.trace_id
        body = message.body

        # Commands take priority
        if body.text.startswith("/"):
            handled = await self._cmd_handler.handle(message)
            if handled:
                logger.debug("event_router_command", trace_id=trace_id)
                return None
            # Unrecognised slash command — fall through to text handler

        # Attachment-based routing
        if body.has_attachment(MediaType.VIDEO_CIRCLE) or body.has_attachment(MediaType.VIDEO):
            session = await self._sessions.get_or_create(
                message.sender.user_id, message.recipient_chat_id
            )
            if session.mode in (TranslationMode.GESTURE_TEXT, TranslationMode.GESTURE_VOICE):
                logger.info("event_router_video", mode=session.mode, trace_id=trace_id)
                return await self._video_handler.handle(message)
            # Wrong mode for video — inform user
            await self._api.send_text(
                message.recipient_chat_id,
                f"⚙️ Видео-кружок получен, но режим сейчас «{session.mode}».\n"
                "Переключись в жестовый режим: /mode gesture_text",
            )
            return None

        if body.has_attachment(MediaType.AUDIO):
            session = await self._sessions.get_or_create(
                message.sender.user_id, message.recipient_chat_id
            )
            if session.mode in (TranslationMode.SPEECH_TEXT, TranslationMode.SPEECH_TTS):
                logger.info("event_router_voice", mode=session.mode, trace_id=trace_id)
                return await self._voice_handler.handle(message)
            # Wrong mode for audio
            await self._api.send_text(
                message.recipient_chat_id,
                f"⚙️ Голосовое получено, но режим сейчас «{session.mode}».\n"
                "Переключись: /mode speech_text",
            )
            return None

        if body.attachments and not body.text:
            await self._api.send_text(message.recipient_chat_id, _UNSUPPORTED_MEDIA)
            return None

        # Plain text (or text + unsupported attachment)
        if body.text.strip():
            logger.info("event_router_text", trace_id=trace_id)
            return await self._text_handler.handle(message)

        logger.debug("event_router_empty_message", trace_id=trace_id)
        return None
