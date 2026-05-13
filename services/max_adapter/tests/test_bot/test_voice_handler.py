"""Tests for VoiceMessageHandler."""

from __future__ import annotations

import pytest

from app.bot.voice_handler import VoiceMessageHandler
from app.domain.entities import TranslationMode
from tests.conftest import FakeMAXAPI, FakePipeline, make_voice_message


@pytest.fixture
def handler(pipeline, session_manager, max_api) -> VoiceMessageHandler:
    return VoiceMessageHandler(pipeline, session_manager, max_api)


class TestVoiceHandlerSpeechText:
    async def test_full_pipeline_speech_text(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        msg = make_voice_message()
        result = await handler.handle(msg)

        pipeline.transcribe.assert_awaited_once()
        pipeline.simplify.assert_awaited_once()
        pipeline.synthesize.assert_not_awaited()
        assert result.asr_text == pipeline.transcribe_result
        assert result.text_output == pipeline.simplify_result
        assert result.audio_bytes is None
        assert len(max_api.sent_texts) == 1
        assert "Распознано" in max_api.sent_texts[0][1]
        assert "Упрощено" in max_api.sent_texts[0][1]

    async def test_asr_text_truncated_in_caption(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        pipeline.transcribe_result = "А" * 300
        msg = make_voice_message()
        await handler.handle(msg)
        caption = max_api.sent_texts[0][1]
        # Caption truncates ASR at 200 chars
        assert len(caption) < 700


class TestVoiceHandlerSpeechTTS:
    async def test_full_pipeline_speech_tts_sends_audio(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TTS)
        msg = make_voice_message()
        result = await handler.handle(msg)

        pipeline.transcribe.assert_awaited_once()
        pipeline.simplify.assert_awaited_once()
        pipeline.synthesize.assert_awaited_once()
        assert result.audio_bytes == pipeline.synthesize_result
        assert len(max_api.sent_audios) == 1
        assert len(max_api.sent_texts) == 0

    async def test_tts_failure_falls_back_to_text(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TTS)
        pipeline.synthesize.side_effect = RuntimeError("TTS unavailable")
        msg = make_voice_message()
        result = await handler.handle(msg)
        # No audio sent, text fallback
        assert result.audio_bytes is None
        assert len(max_api.sent_texts) == 1


class TestVoiceHandlerErrors:
    async def test_missing_audio_attachment(
        self, handler: VoiceMessageHandler, max_api: FakeMAXAPI, session_manager
    ) -> None:
        from app.domain.entities import MAXMessage, MAXMessageBody, MAXUser
        msg = MAXMessage(
            sender=MAXUser(user_id=42, first_name="T", username=None, is_bot=False),
            recipient_chat_id=100,
            body=MAXMessageBody(mid="x", text="", attachments=[]),
            trace_id="err-001",
        )
        result = await handler.handle(msg)
        assert result.latency_ms == 0
        assert any("Аудио не найдено" in t for _, t in max_api.sent_texts)

    async def test_download_failure_sends_error(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        max_api.download_media.side_effect = ConnectionError("CDN unreachable")
        msg = make_voice_message()
        await handler.handle(msg)
        assert any("скачать" in t for _, t in max_api.sent_texts)

    async def test_asr_failure_sends_fallback(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        pipeline.transcribe.side_effect = RuntimeError("ASR down")
        msg = make_voice_message()
        await handler.handle(msg)
        assert any("распознать" in t.lower() for _, t in max_api.sent_texts)

    async def test_empty_asr_result_sends_hint(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        pipeline.transcribe_result = "   "
        msg = make_voice_message()
        await handler.handle(msg)
        assert any("Речь не распознана" in t for _, t in max_api.sent_texts)

    async def test_nlp_failure_falls_back_to_asr_text(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        pipeline.simplify.side_effect = RuntimeError("NLP down")
        msg = make_voice_message()
        result = await handler.handle(msg)
        # Falls back to raw ASR text
        assert result.text_output == pipeline.transcribe_result

    async def test_from_pipeline_field(
        self, handler: VoiceMessageHandler, pipeline: FakePipeline,
        max_api: FakeMAXAPI, session_manager
    ) -> None:
        await session_manager.set_mode(42, 100, TranslationMode.SPEECH_TEXT)
        msg = make_voice_message()
        result = await handler.handle(msg)
        assert "asr" in result.from_pipeline
        assert "nlp" in result.from_pipeline
        assert "tts" not in result.from_pipeline
