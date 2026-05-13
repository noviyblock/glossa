"""Shared test fixtures for MAX adapter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.entities import (
    MAXAttachment,
    MAXMessage,
    MAXMessageBody,
    MAXUser,
    MediaType,
    TranslationMode,
    UserSession,
)
from app.domain.interfaces import MAXAPIPort, PipelineClientPort
from app.session.session_manager import InMemorySessionStore, SessionManager


# ── Fake pipeline client ──────────────────────────────────────────────────────

class FakePipeline(PipelineClientPort):
    def __init__(self) -> None:
        self.transcribe_result: str = "привет мир"
        self.chat_result: str = "Ответ бота"
        self.translate_gloss_result: str = "Привет, как дела?"
        self.simplify_result: str = "Упрощённый текст"
        self.synthesize_result: bytes = b"\x00" * 1024  # fake WAV
        self.recognize_gesture_result: str = "ПРИВЕТ КАК ДЕЛА"

        self.transcribe = AsyncMock(side_effect=self._transcribe)
        self.chat = AsyncMock(side_effect=self._chat)
        self.translate_gloss = AsyncMock(side_effect=self._translate_gloss)
        self.simplify = AsyncMock(side_effect=self._simplify)
        self.synthesize = AsyncMock(side_effect=self._synthesize)
        self.recognize_gesture = AsyncMock(side_effect=self._recognize_gesture)

    async def _transcribe(self, audio_bytes, session_id, language="ru") -> str:
        return self.transcribe_result

    async def _chat(self, text, session_id, domain="general", history="") -> str:
        return self.chat_result

    async def _translate_gloss(self, gloss_sequence, session_id, domain="general") -> str:
        return self.translate_gloss_result

    async def _simplify(self, text, session_id, domain="medical") -> str:
        return self.simplify_result

    async def _synthesize(self, text, voice_id="aidar", sample_rate=24000) -> bytes:
        return self.synthesize_result

    async def _recognize_gesture(self, video_bytes, session_id) -> str:
        return self.recognize_gesture_result


# ── Fake MAX API ──────────────────────────────────────────────────────────────

class FakeMAXAPI(MAXAPIPort):
    def __init__(self) -> None:
        self.sent_texts: list[tuple[int, str]] = []
        self.sent_audios: list[tuple[int, bytes]] = []
        self.typing_sent: list[int] = []
        self.download_result: bytes = b"\x00" * 512

        self.send_text = AsyncMock(side_effect=self._send_text)
        self.send_audio = AsyncMock(side_effect=self._send_audio)
        self.send_typing = AsyncMock(side_effect=self._send_typing)
        self.download_media = AsyncMock(side_effect=self._download_media)

    async def _send_text(self, chat_id, text, **_) -> None:
        self.sent_texts.append((chat_id, text))

    async def _send_audio(self, chat_id, audio_bytes, **_) -> None:
        self.sent_audios.append((chat_id, audio_bytes))

    async def _send_typing(self, chat_id) -> None:
        self.typing_sent.append(chat_id)

    async def _download_media(self, url) -> bytes:
        return self.download_result


# ── Message builders ──────────────────────────────────────────────────────────

def make_user(user_id: int = 42, name: str = "Test User") -> MAXUser:
    parts = name.split(" ", 1)
    return MAXUser(
        user_id=user_id,
        first_name=parts[0],
        last_name=parts[1] if len(parts) > 1 else None,
        username="testuser",
        is_bot=False,
    )


def make_text_message(text: str, chat_id: int = 100, user_id: int = 42) -> MAXMessage:
    return MAXMessage(
        sender=make_user(user_id),
        recipient_chat_id=chat_id,
        body=MAXMessageBody(mid="msg-001", text=text, attachments=[]),
        trace_id="tst-0001",
    )


def make_voice_message(chat_id: int = 100, user_id: int = 42) -> MAXMessage:
    attachment = MAXAttachment(
        type=MediaType.AUDIO,
        payload={"url": "https://cdn.max.ru/audio/test.ogg", "token": "tok123"},
    )
    return MAXMessage(
        sender=make_user(user_id),
        recipient_chat_id=chat_id,
        body=MAXMessageBody(mid="msg-002", text="", attachments=[attachment]),
        trace_id="tst-0002",
    )


def make_video_message(chat_id: int = 100, user_id: int = 42) -> MAXMessage:
    attachment = MAXAttachment(
        type=MediaType.VIDEO_CIRCLE,
        payload={"url": "https://cdn.max.ru/video/test.mp4", "token": "tok456"},
    )
    return MAXMessage(
        sender=make_user(user_id),
        recipient_chat_id=chat_id,
        body=MAXMessageBody(mid="msg-003", text="", attachments=[attachment]),
        trace_id="tst-0003",
    )


# ── Session manager fixture ───────────────────────────────────────────────────

@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(InMemorySessionStore())


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def max_api() -> FakeMAXAPI:
    return FakeMAXAPI()
