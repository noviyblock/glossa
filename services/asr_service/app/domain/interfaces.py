"""Abstract ports for the ASR domain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np

from .entities import (
    ASRSession,
    FinalTranscript,
    PartialTranscript,
    TranscriptSegment,
    VADSegment,
)


class VADPort(ABC):
    """Voice Activity Detector."""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> list[VADSegment]:
        """Return detected speech segments from a chunk of audio."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (call between utterances / sessions)."""


class TranscriberPort(ABC):
    """Speech-to-text transcriber."""

    @abstractmethod
    async def ensure_loaded(self) -> None:
        """Load model weights if not already loaded."""

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
        beam_size: int,
    ) -> list[TranscriptSegment]:
        """Transcribe audio and return all segments."""

    @abstractmethod
    async def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
        beam_size: int,
    ) -> AsyncIterator[TranscriptSegment]:
        """Yield segments as they are decoded."""

    @abstractmethod
    async def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (language_code, probability) from first 30 s of audio."""

    @abstractmethod
    async def warmup(self, iterations: int = 3) -> float:
        """Return mean warmup latency ms."""


class MedicalAdapterPort(ABC):
    """Post-processing hook for medical/domain terminology."""

    @abstractmethod
    def adapt(self, text: str, language: str) -> tuple[str, list[str]]:
        """Return (corrected_text, list_of_found_terms)."""

    @abstractmethod
    def add_term(self, term: str, correction: str, language: str = "ru") -> None:
        """Register a new term mapping at runtime."""


class SessionStorePort(ABC):
    """Session state storage."""

    @abstractmethod
    async def create(self, session: ASRSession) -> None: ...

    @abstractmethod
    async def get(self, session_id: str) -> ASRSession | None: ...

    @abstractmethod
    async def get_by_token(self, token: str) -> ASRSession | None: ...

    @abstractmethod
    async def update(self, session: ASRSession) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...


class AudioBufferPort(ABC):
    """Rolling audio accumulation buffer."""

    @abstractmethod
    def push(self, audio: np.ndarray) -> None: ...

    @abstractmethod
    def flush(self) -> np.ndarray: ...

    @abstractmethod
    def peek(self, n_samples: int) -> np.ndarray: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def n_samples(self) -> int: ...
