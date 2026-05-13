"""Tests for VideoMessageHandler."""

from __future__ import annotations

import pytest

from app.bot.video_handler import VideoMessageHandler
from app.domain.entities import TranslationMode
from tests.conftest import FakeMAXAPI, FakePipeline, make_video_message


@pytest.fixture
def handler(pipeline, session_manager, max_api) -> VideoMessageHandler:
    return VideoMessageHandler(pipeline, session_manager, max_api)


class TestVideoHandlerGestureText:
    async def test_full_pipeline_gesture_text(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        msg = make_video_message()
        result = await handler.handle(msg)

        pipeline.recognize_gesture.assert_awaited_once()
        pipeline.translate_gloss.assert_awaited_once()
        pipeline.synthesize.assert_not_awaited()
        assert result.gloss_sequence == pipeline.recognize_gesture_result
        assert result.text_output == pipeline.translate_gloss_result
        assert result.audio_bytes is None
        assert len(max_api.sent_texts) == 1
        assert "Глосс" in max_api.sent_texts[0][1]
        assert "Перевод" in max_api.sent_texts[0][1]


class TestVideoHandlerGestureVoice:
    async def test_full_pipeline_gesture_voice_sends_audio(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_VOICE)
        msg = make_video_message()
        result = await handler.handle(msg)

        pipeline.recognize_gesture.assert_awaited_once()
        pipeline.translate_gloss.assert_awaited_once()
        pipeline.synthesize.assert_awaited_once()
        assert result.audio_bytes == pipeline.synthesize_result
        assert len(max_api.sent_audios) == 1
        assert len(max_api.sent_texts) == 0

    async def test_tts_failure_falls_back_to_text(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_VOICE)
        pipeline.synthesize.side_effect = RuntimeError("TTS down")
        msg = make_video_message()
        result = await handler.handle(msg)
        assert result.audio_bytes is None
        assert len(max_api.sent_texts) == 1


class TestVideoHandlerErrors:
    async def test_missing_video_attachment(
        self, handler: VideoMessageHandler, max_api: FakeMAXAPI
    ) -> None:
        from app.domain.entities import MAXMessage, MAXMessageBody, MAXUser
        msg = MAXMessage(
            sender=MAXUser(user_id=42, first_name="T", username=None, is_bot=False),
            recipient_chat_id=100,
            body=MAXMessageBody(mid="x", text="", attachments=[]),
            trace_id="err-002",
        )
        result = await handler.handle(msg)
        assert result.latency_ms == 0
        assert any("Видео не найдено" in t for _, t in max_api.sent_texts)

    async def test_download_failure(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        max_api.download_media.side_effect = ConnectionError("CDN error")
        msg = make_video_message()
        await handler.handle(msg)
        assert any("скачать" in t for _, t in max_api.sent_texts)

    async def test_cv_failure_sends_fallback(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        pipeline.recognize_gesture.side_effect = RuntimeError("CV down")
        msg = make_video_message()
        await handler.handle(msg)
        assert any("распознан" in t.lower() for _, t in max_api.sent_texts)

    async def test_empty_gloss_sends_hint(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        pipeline.recognize_gesture_result = ""
        msg = make_video_message()
        await handler.handle(msg)
        assert any("распознан" in t.lower() for _, t in max_api.sent_texts)

    async def test_nlp_failure_falls_back_to_gloss(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_TEXT)
        pipeline.translate_gloss.side_effect = RuntimeError("NLP down")
        msg = make_video_message()
        result = await handler.handle(msg)
        assert result.text_output == pipeline.recognize_gesture_result

    async def test_from_pipeline_field_cv_nlp_tts(
        self, handler: VideoMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.GESTURE_VOICE)
        msg = make_video_message()
        result = await handler.handle(msg)
        assert "cv" in result.from_pipeline
        assert "nlp" in result.from_pipeline
        assert "tts" in result.from_pipeline
