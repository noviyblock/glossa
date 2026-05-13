"""Abstract ports for TTS domain — implementation-independent contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import numpy as np

from .entities import (
    AudioChunk,
    AudioFormat,
    BenchmarkResult,
    PhraseCacheEntry,
    SynthesisRequest,
    SynthesisResult,
    VoiceProfile,
)


class SynthesizerPort(ABC):
    """Core synthesis engine — Silero today, XTTS tomorrow."""

    @abstractmethod
    async def ensure_loaded(self) -> None: ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        put_accent: bool = True,
        put_yo: bool = True,
    ) -> np.ndarray:
        """Return float32 PCM at the voice's sample_rate."""

    @abstractmethod
    async def synthesize_sentences(
        self,
        sentences: list[str],
        voice: VoiceProfile,
        put_accent: bool = True,
        put_yo: bool = True,
    ) -> AsyncIterator[tuple[str, np.ndarray]]:
        """Yield (sentence, pcm) pairs for streaming."""

    @abstractmethod
    def list_voices(self) -> list[VoiceProfile]: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def benchmark(self, text: str, voice: VoiceProfile, n: int) -> BenchmarkResult: ...


class CachePort(ABC):
    """Phrase audio cache."""

    @abstractmethod
    async def get(self, key: str) -> PhraseCacheEntry | None: ...

    @abstractmethod
    async def set(self, entry: PhraseCacheEntry) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def stats(self) -> dict: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class AudioEncoderPort(ABC):
    """PCM → encoded bytes."""

    @abstractmethod
    def encode(self, pcm: np.ndarray, sample_rate: int, fmt: AudioFormat) -> bytes: ...

    @abstractmethod
    def pcm_to_wav(self, pcm: np.ndarray, sample_rate: int) -> bytes: ...

    @abstractmethod
    def pcm_to_ogg(self, pcm: np.ndarray, sample_rate: int) -> bytes: ...
