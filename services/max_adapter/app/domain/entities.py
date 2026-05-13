"""MAX Messenger bot domain entities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


# ── Translation modes ─────────────────────────────────────────────────────────

class TranslationMode(StrEnum):
    GESTURE_TEXT  = "gesture_text"   # video circle → gloss → Russian text
    GESTURE_VOICE = "gesture_voice"  # video circle → gloss → text → TTS audio
    SPEECH_TEXT   = "speech_text"    # voice message → ASR → simplified text
    SPEECH_TTS    = "speech_tts"     # voice message → ASR → text → TTS audio
    TEXT_RESPONSE = "text_response"  # text → NLP chat response


class MediaType(StrEnum):
    AUDIO        = "audio"
    VIDEO        = "video"
    VIDEO_CIRCLE = "video_circle"
    IMAGE        = "image"
    FILE         = "file"
    STICKER      = "sticker"


class UpdateType(StrEnum):
    MESSAGE_CREATED   = "message_created"
    MESSAGE_EDITED    = "message_edited"
    MESSAGE_REMOVED   = "message_removed"
    MESSAGE_CALLBACK  = "message_callback"
    BOT_ADDED         = "bot_added"
    BOT_STARTED       = "bot_started"
    BOT_STOPPED       = "bot_stopped"
    BOT_REMOVED       = "bot_removed"
    CHAT_TITLE_CHANGED = "chat_title_changed"
    DIALOG_CLEARED    = "dialog_cleared"
    DIALOG_MUTED      = "dialog_muted"
    DIALOG_UNMUTED    = "dialog_unmuted"
    DIALOG_REMOVED    = "dialog_removed"
    USER_ADDED        = "user_added"
    USER_REMOVED      = "user_removed"


# ── MAX message model ─────────────────────────────────────────────────────────

@dataclass
class MAXUser:
    user_id: int
    first_name: str
    last_name: str | None = None
    username: str | None = None
    is_bot: bool = False

    @property
    def name(self) -> str:
        """Full display name."""
        parts = [self.first_name]
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts)


@dataclass
class MAXAttachment:
    type: MediaType
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str | None:
        return self.payload.get("url")

    @property
    def token(self) -> str | None:
        return self.payload.get("token")

    @property
    def file_id(self) -> str | None:
        return self.payload.get("id") or self.payload.get("file_id")


@dataclass
class MAXMessageBody:
    mid: str
    seq: int = 0
    text: str = ""
    attachments: list[MAXAttachment] = field(default_factory=list)

    def has_attachment(self, media_type: MediaType) -> bool:
        return any(a.type == media_type for a in self.attachments)

    def get_attachment(self, media_type: MediaType) -> MAXAttachment | None:
        for a in self.attachments:
            if a.type == media_type:
                return a
        return None


@dataclass
class MAXMessage:
    sender: MAXUser
    recipient_chat_id: int
    body: MAXMessageBody
    recipient_user_id: int | None = None
    timestamp: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class MAXUpdate:
    update_type: UpdateType
    message: MAXMessage | None = None
    timestamp: float = field(default_factory=time.time)


# ── User session ──────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str            # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    mode: TranslationMode = TranslationMode.TEXT_RESPONSE


@dataclass
class UserSession:
    user_id: int
    chat_id: int
    mode: TranslationMode = TranslationMode.TEXT_RESPONSE
    language: str = "ru"
    domain: str = "general"       # general | medical | banking
    conversation: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_requests: int = 0
    session_id: str = field(default_factory=lambda: str(uuid4()))

    def add_turn(self, role: str, content: str) -> None:
        self.conversation.append(ConversationTurn(role=role, content=content, mode=self.mode))
        # Keep last 20 turns to limit context size
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-20:]
        self.updated_at = time.time()

    def history_text(self, n: int = 6) -> str:
        recent = self.conversation[-n:]
        return "\n".join(
            f"{'Пользователь' if t.role == 'user' else 'Ассистент'}: {t.content}"
            for t in recent
        )


# ── Pipeline results (internal) ───────────────────────────────────────────────

@dataclass
class TranslationResult:
    trace_id: str
    mode: TranslationMode
    text_output: str = ""
    audio_bytes: bytes | None = None
    audio_format: str = "wav"
    latency_ms: float = 0.0
    asr_text: str = ""           # intermediate ASR transcript
    gloss_sequence: str = ""     # intermediate gloss sequence
    confidence: float = 0.0
    from_pipeline: str = ""      # which pipeline produced this


# ── Rate limit state ──────────────────────────────────────────────────────────

@dataclass
class RateLimitState:
    user_id: int
    window_start: float
    count: int = 0

    def increment(self) -> None:
        self.count += 1

    def is_exceeded(self, limit: int) -> bool:
        return self.count >= limit
