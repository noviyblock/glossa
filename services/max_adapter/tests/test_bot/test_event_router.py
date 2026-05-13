"""Tests for EventRouter dispatch logic."""

from __future__ import annotations

import pytest

from app.bot.event_router import EventRouter
from app.domain.entities import (
    MAXAttachment,
    MAXMessageBody,
    MediaType,
    TranslationMode,
    UpdateType,
)
from tests.conftest import (
    FakeMAXAPI,
    FakePipeline,
    make_text_message,
    make_video_message,
    make_voice_message,
)


@pytest.fixture
def router(pipeline, session_manager, max_api) -> EventRouter:
    return EventRouter(pipeline=pipeline, session_manager=session_manager, max_api=max_api)


class TestEventRouterText:
    async def test_routes_plain_text_to_text_handler(
        self, router: EventRouter, max_api: FakeMAXAPI, pipeline: FakePipeline
    ) -> None:
        msg = make_text_message("Привет, как дела?")
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is not None
        assert result.text_output == pipeline.chat_result
        assert len(max_api.sent_texts) == 1

    async def test_ignores_non_message_created(
        self, router: EventRouter, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("test")
        result = await router.route(UpdateType.BOT_ADDED, msg)
        assert result is None
        assert len(max_api.sent_texts) == 0

    async def test_empty_text_returns_none(
        self, router: EventRouter, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("")
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is None


class TestEventRouterCommands:
    async def test_start_command_handled(
        self, router: EventRouter, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/start")
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is None
        assert len(max_api.sent_texts) == 1
        assert "Глосса" in max_api.sent_texts[0][1]

    async def test_mode_command_changes_mode(
        self, router: EventRouter, max_api: FakeMAXAPI, session_manager
    ) -> None:
        msg = make_text_message("/mode speech_text")
        await router.route(UpdateType.MESSAGE_CREATED, msg)
        session = await session_manager.get_or_create(42, 100)
        assert session.mode == TranslationMode.SPEECH_TEXT

    async def test_unknown_command_falls_through_to_text(
        self, router: EventRouter, pipeline: FakePipeline, max_api: FakeMAXAPI
    ) -> None:
        msg = make_text_message("/unknown_cmd")
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        # Unknown command → handled as text → pipeline.chat called
        pipeline.chat.assert_awaited()


class TestEventRouterVoice:
    async def test_routes_audio_in_speech_text_mode(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        pipeline: FakePipeline,
        session_manager,
    ) -> None:
        # Set speech_text mode first
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        msg = make_voice_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is not None
        pipeline.transcribe.assert_awaited_once()
        pipeline.simplify.assert_awaited_once()
        assert len(max_api.sent_texts) == 1

    async def test_routes_audio_in_speech_tts_mode_sends_audio(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        pipeline: FakePipeline,
        session_manager,
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TTS)
        msg = make_voice_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is not None
        pipeline.synthesize.assert_awaited_once()
        assert len(max_api.sent_audios) == 1

    async def test_audio_in_wrong_mode_sends_hint(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        session_manager,
    ) -> None:
        # Default mode is TEXT_RESPONSE — audio should get a hint
        msg = make_voice_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is None
        assert any("/mode speech_text" in t for _, t in max_api.sent_texts)


class TestEventRouterVideo:
    async def test_routes_video_circle_in_gesture_text_mode(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        pipeline: FakePipeline,
        session_manager,
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        msg = make_video_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is not None
        pipeline.recognize_gesture.assert_awaited_once()
        pipeline.translate_gloss.assert_awaited_once()
        assert len(max_api.sent_texts) == 1

    async def test_routes_video_in_gesture_voice_mode_sends_audio(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        pipeline: FakePipeline,
        session_manager,
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_VOICE)
        msg = make_video_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is not None
        pipeline.synthesize.assert_awaited_once()
        assert len(max_api.sent_audios) == 1

    async def test_video_in_wrong_mode_sends_hint(
        self,
        router: EventRouter,
        max_api: FakeMAXAPI,
        session_manager,
    ) -> None:
        msg = make_video_message()
        result = await router.route(UpdateType.MESSAGE_CREATED, msg)
        assert result is None
        assert any("gesture_text" in t for _, t in max_api.sent_texts)
