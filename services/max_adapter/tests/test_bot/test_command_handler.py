"""Tests for CommandHandler."""

from __future__ import annotations

import pytest

from app.bot.command_handler import CommandHandler
from app.domain.entities import TranslationMode
from tests.conftest import FakeMAXAPI, make_text_message


@pytest.fixture
def handler(session_manager, max_api) -> CommandHandler:
    return CommandHandler(session_manager, max_api)


class TestCommandHandlerRouting:
    async def test_non_command_returns_false(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("hello world")
        handled = await handler.handle(msg)
        assert handled is False
        assert len(max_api.sent_texts) == 0

    async def test_unknown_command_returns_false(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/nonexistent_command")
        handled = await handler.handle(msg)
        assert handled is False

    async def test_command_with_bot_suffix(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/start@GlossaBot")
        handled = await handler.handle(msg)
        assert handled is True
        assert any("Глосса" in t for _, t in max_api.sent_texts)


class TestStartHelp:
    async def test_start_sends_welcome(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/start")
        await handler.handle(msg)
        assert len(max_api.sent_texts) == 1
        welcome = max_api.sent_texts[0][1]
        assert "Глосса" in welcome
        assert "/mode" in welcome
        assert "/help" in welcome

    async def test_help_sends_help(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/help")
        await handler.handle(msg)
        text = max_api.sent_texts[0][1]
        assert "speech_text" in text
        assert "gesture_text" in text
        assert "/domain" in text


class TestModeCommand:
    async def test_mode_without_arg_lists_modes(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/mode")
        await handler.handle(msg)
        text = max_api.sent_texts[0][1]
        assert "gesture_text" in text
        assert "speech_tts" in text

    @pytest.mark.parametrize("arg,expected_mode", [
        ("gesture_text",  TranslationMode.GESTURE_TEXT),
        ("gesture_voice", TranslationMode.GESTURE_VOICE),
        ("speech_text",   TranslationMode.SPEECH_TEXT),
        ("speech_tts",    TranslationMode.SPEECH_TTS),
        ("text",          TranslationMode.TEXT_RESPONSE),
        ("text_response", TranslationMode.TEXT_RESPONSE),
    ])
    async def test_mode_sets_correct_mode(
        self, handler: CommandHandler, max_api: FakeMAXAPI,
        session_manager, arg: str, expected_mode: TranslationMode
    ) -> None:
        msg = make_text_message(f"/mode {arg}")
        await handler.handle(msg)
        session = await session_manager.get_or_create(42, 100)
        assert session.mode == expected_mode
        assert "✅" in max_api.sent_texts[0][1]

    async def test_invalid_mode_sends_error(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/mode flying_unicorn")
        await handler.handle(msg)
        assert "❌" in max_api.sent_texts[0][1]

    async def test_mode_case_insensitive(
        self, handler: CommandHandler, max_api: FakeMAXAPI, session_manager
    ) -> None:
        msg = make_text_message("/mode SPEECH_TEXT")
        await handler.handle(msg)
        session = await session_manager.get_or_create(42, 100)
        assert session.mode == TranslationMode.SPEECH_TEXT


class TestDomainCommand:
    @pytest.mark.parametrize("domain", ["general", "medical", "banking"])
    async def test_valid_domains(
        self, handler: CommandHandler, max_api: FakeMAXAPI,
        session_manager, domain: str
    ) -> None:
        msg = make_text_message(f"/domain {domain}")
        await handler.handle(msg)
        session = await session_manager.get_or_create(42, 100)
        assert session.domain == domain
        assert "✅" in max_api.sent_texts[0][1]

    async def test_invalid_domain_sends_hint(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/domain astrology")
        await handler.handle(msg)
        text = max_api.sent_texts[0][1]
        assert "general" in text
        assert "medical" in text

    async def test_domain_without_arg_sends_hint(
        self, handler: CommandHandler, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/domain")
        await handler.handle(msg)
        assert "general" in max_api.sent_texts[0][1]


class TestStatusCommand:
    async def test_status_shows_current_settings(
        self, handler: CommandHandler, max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TTS)
        await session_manager.set_domain(42, 100, "medical")
        msg = make_text_message("/status")
        await handler.handle(msg)
        text = max_api.sent_texts[0][1]
        assert "medical" in text
        assert "Запросов" in text


class TestClearCommand:
    async def test_clear_resets_conversation(
        self, handler: CommandHandler, max_api: FakeMAXAPI, session_manager
    ) -> None:
        session = await session_manager.get_or_create(42, 100)
        session.add_turn("user", "Привет")
        await session_manager.save(session)

        msg = make_text_message("/clear")
        await handler.handle(msg)
        refreshed = await session_manager.get_or_create(42, 100)
        assert len(refreshed.conversation) == 0
        assert "очищена" in max_api.sent_texts[0][1]
