"""Bot command handler — /start, /help, /mode, /domain, /status, /clear."""

from __future__ import annotations

import textwrap

from glossa_common.logging import get_logger

from ..domain.entities import MAXMessage, TranslationMode
from ..domain.interfaces import MAXAPIPort
from ..session.session_manager import SessionManager

logger = get_logger(__name__)

# ── Mode descriptions ─────────────────────────────────────────────────────────

_MODE_DESCRIPTIONS = {
    TranslationMode.GESTURE_TEXT:  "🤟 Жест → Текст (видео-кружок → распознавание жестов → русский текст)",
    TranslationMode.GESTURE_VOICE: "🤟🔊 Жест → Голос (видео-кружок → распознавание → TTS озвучка)",
    TranslationMode.SPEECH_TEXT:   "🎙️ Речь → Текст (голосовое сообщение → расшифровка → упрощение)",
    TranslationMode.SPEECH_TTS:    "🎙️🔊 Речь → Голос (голосовое → расшифровка → TTS переозвучка)",
    TranslationMode.TEXT_RESPONSE: "💬 Текст → Ответ (текстовое сообщение → умный ответ)",
}

_MODE_COMMANDS = {
    "gesture_text":  TranslationMode.GESTURE_TEXT,
    "gesture_voice": TranslationMode.GESTURE_VOICE,
    "speech_text":   TranslationMode.SPEECH_TEXT,
    "speech_tts":    TranslationMode.SPEECH_TTS,
    "text":          TranslationMode.TEXT_RESPONSE,
    "text_response": TranslationMode.TEXT_RESPONSE,
}

_DOMAIN_COMMANDS = {"general", "medical", "banking"}

_WELCOME = textwrap.dedent("""\
    👋 Привет! Я Глосса — AI-переводчик русского жестового языка (РЖЯ).

    Я умею:
    • Распознавать жесты из видео-кружков
    • Расшифровывать голосовые сообщения
    • Упрощать медицинские и банковские тексты
    • Отвечать на вопросы в текстовом чате

    Команды:
    /mode — сменить режим перевода
    /domain — выбрать предметную область
    /status — текущие настройки
    /clear — очистить историю диалога
    /help — подробная справка
""")

_HELP = textwrap.dedent("""\
    📖 Справка по режимам:

    /mode gesture_text   — видео → текст
    /mode gesture_voice  — видео → голос
    /mode speech_text    — голос → текст
    /mode speech_tts     — голос → голос
    /mode text           — текст → ответ

    /domain general   — общая тематика
    /domain medical   — медицина
    /domain banking   — банковское дело

    /clear — сбросить контекст диалога
    /status — показать текущие настройки

    Примеры использования:
    1. Отправьте видео-кружок с жестом РЖЯ
    2. Отправьте голосовое сообщение на русском
    3. Напишите вопрос текстом
""")


class CommandHandler:
    def __init__(self, session_manager: SessionManager, max_api: MAXAPIPort) -> None:
        self._sessions = session_manager
        self._api = max_api

    async def handle(self, message: MAXMessage) -> bool:
        """Return True if the message was a command and was handled."""
        text = message.body.text.strip()
        if not text.startswith("/"):
            return False

        parts = text.split(maxsplit=1)
        command = parts[0].lower().lstrip("/").split("@")[0]  # strip @botname suffix
        arg = parts[1].strip() if len(parts) > 1 else ""

        user_id = message.sender.user_id
        chat_id = message.recipient_chat_id

        handler = {
            "start":   self._cmd_start,
            "help":    self._cmd_help,
            "mode":    self._cmd_mode,
            "domain":  self._cmd_domain,
            "status":  self._cmd_status,
            "clear":   self._cmd_clear,
        }.get(command)

        if handler is None:
            return False

        logger.info("bot_command", command=command, user_id=user_id, arg=arg)
        await handler(user_id, chat_id, arg, message)
        return True

    # ── Command implementations ───────────────────────────────────────────────

    async def _cmd_start(self, user_id: int, chat_id: int, _arg: str, _msg: MAXMessage) -> None:
        await self._sessions.get_or_create(user_id, chat_id)
        await self._api.send_text(chat_id, _WELCOME)

    async def _cmd_help(self, user_id: int, chat_id: int, _arg: str, _msg: MAXMessage) -> None:
        await self._api.send_text(chat_id, _HELP)

    async def _cmd_mode(self, user_id: int, chat_id: int, arg: str, _msg: MAXMessage) -> None:
        if not arg:
            mode_list = "\n".join(
                f"  /mode {k} — {v}" for k, v in _MODE_DESCRIPTIONS.items()
            )
            await self._api.send_text(
                chat_id,
                f"Текущие режимы:\n{mode_list}\n\nИспользуй: /mode <название>",
            )
            return

        mode = _MODE_COMMANDS.get(arg.lower())
        if mode is None:
            await self._api.send_text(
                chat_id,
                f"❌ Неизвестный режим: «{arg}». Используй /help для списка.",
            )
            return

        session = await self._sessions.set_mode(user_id, chat_id, mode)
        desc = _MODE_DESCRIPTIONS[mode]
        await self._api.send_text(chat_id, f"✅ Режим установлен: {desc}")

    async def _cmd_domain(self, user_id: int, chat_id: int, arg: str, _msg: MAXMessage) -> None:
        if not arg or arg.lower() not in _DOMAIN_COMMANDS:
            await self._api.send_text(
                chat_id,
                "Доступные области: general, medical, banking\nПример: /domain medical",
            )
            return
        await self._sessions.set_domain(user_id, chat_id, arg.lower())
        await self._api.send_text(chat_id, f"✅ Область: {arg.lower()}")

    async def _cmd_status(self, user_id: int, chat_id: int, _arg: str, _msg: MAXMessage) -> None:
        session = await self._sessions.get_or_create(user_id, chat_id)
        mode_desc = _MODE_DESCRIPTIONS.get(session.mode, session.mode)
        text = (
            f"📊 Настройки:\n"
            f"Режим: {mode_desc}\n"
            f"Область: {session.domain}\n"
            f"Язык: {session.language}\n"
            f"Запросов: {session.total_requests}\n"
            f"Поворотов диалога: {len(session.conversation)}"
        )
        await self._api.send_text(chat_id, text)

    async def _cmd_clear(self, user_id: int, chat_id: int, _arg: str, _msg: MAXMessage) -> None:
        await self._sessions.clear_context(user_id, chat_id)
        await self._api.send_text(chat_id, "🗑️ История диалога очищена.")
