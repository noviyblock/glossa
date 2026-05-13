"""Abstract ports for the MAX bot domain."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import MAXUpdate, TranslationResult, UserSession


class SessionStorePort(ABC):
    @abstractmethod
    async def get(self, user_id: int) -> UserSession | None: ...

    @abstractmethod
    async def save(self, session: UserSession) -> None: ...

    @abstractmethod
    async def delete(self, user_id: int) -> None: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class PipelineClientPort(ABC):
    """Internal Glossa service pipeline — ASR + NLP + TTS + CV."""

    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, session_id: str, language: str = "ru"
    ) -> str: ...

    @abstractmethod
    async def chat(
        self, text: str, session_id: str, domain: str = "general", history: str = ""
    ) -> str: ...

    @abstractmethod
    async def translate_gloss(
        self, gloss_sequence: str, session_id: str, domain: str = "general"
    ) -> str: ...

    @abstractmethod
    async def simplify(
        self, text: str, session_id: str, domain: str = "medical"
    ) -> str: ...

    @abstractmethod
    async def synthesize(
        self, text: str, voice_id: str = "aidar", sample_rate: int = 24000
    ) -> bytes: ...

    @abstractmethod
    async def recognize_gesture(
        self, video_bytes: bytes, session_id: str
    ) -> str: ...


class MAXAPIPort(ABC):
    """MAX Messenger Bot API client."""

    @abstractmethod
    async def send_text(self, chat_id: int, text: str, trace_id: str = "") -> None: ...

    @abstractmethod
    async def send_audio(
        self, chat_id: int, audio_bytes: bytes, caption: str = "", trace_id: str = ""
    ) -> None: ...

    @abstractmethod
    async def send_typing(self, chat_id: int) -> None: ...

    @abstractmethod
    async def download_media(self, url: str) -> bytes: ...

    @abstractmethod
    async def get_bot_info(self) -> dict: ...
