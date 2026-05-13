"""Shared fixtures for the TTS service test suite."""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.audio.encoder import AudioEncoder
from app.cache.phrase_cache import InMemoryPhraseCache
from app.domain.entities import (
    AudioFormat,
    PhraseCacheEntry,
    SynthesisRequest,
    VoiceProfile,
)
from app.domain.text_processor import TextPreprocessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pcm(duration_s: float = 0.5, sample_rate: int = 24000) -> np.ndarray:
    """Silent float32 PCM array."""
    n = int(duration_s * sample_rate)
    return np.zeros(n, dtype=np.float32)


def make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 24000) -> bytes:
    encoder = AudioEncoder()
    return encoder.pcm_to_wav(make_pcm(duration_s, sample_rate), sample_rate)


def make_voice(voice_id: str = "aidar") -> VoiceProfile:
    return VoiceProfile(
        id=voice_id,
        display_name="Test Voice",
        speaker_key=voice_id,
        language="ru",
        gender="male",
        engine="silero",
        sample_rate=24000,
    )


def make_cache_entry(
    key: str = "testkey",
    text: str = "Привет мир.",
    fmt: AudioFormat = AudioFormat.WAV,
) -> PhraseCacheEntry:
    wav = make_wav_bytes()
    return PhraseCacheEntry(
        key=key,
        audio_bytes=wav,
        format=fmt,
        sample_rate=24000,
        duration_s=0.5,
        char_count=len(text),
    )


# ── Fake synthesizer ──────────────────────────────────────────────────────────

class FakeSynthesizer:
    """Returns deterministic silent PCM without loading any model."""

    def __init__(self, duration_s: float = 0.5) -> None:
        self._duration_s = duration_s
        self.calls: list[dict] = []

    async def ensure_loaded(self) -> None:
        pass

    async def warmup(self) -> None:
        pass

    async def synthesize(
        self, text: str, voice: VoiceProfile, put_accent: bool = True, put_yo: bool = True
    ) -> np.ndarray:
        self.calls.append({"text": text, "voice": voice.id})
        return make_pcm(self._duration_s, voice.sample_rate)

    async def synthesize_sentences(
        self, sentences: list[str], voice: VoiceProfile, **kwargs
    ) -> AsyncIterator[tuple[str, np.ndarray]]:
        for s in sentences:
            yield s, make_pcm(self._duration_s, voice.sample_rate)

    def list_voices(self) -> list[VoiceProfile]:
        return [make_voice("aidar"), make_voice("baya")]

    def get_voice(self, voice_id: str) -> VoiceProfile:
        return make_voice(voice_id)

    async def benchmark(self, text: str, voice: VoiceProfile, n: int = 10):
        from app.domain.entities import BenchmarkResult
        return BenchmarkResult(
            n_runs=n, text_len=len(text), voice_id=voice.id,
            sample_rate=voice.sample_rate,
            p50_ms=50.0, p95_ms=80.0, p99_ms=100.0,
            min_ms=30.0, max_ms=120.0, rtf=0.002, chars_per_sec=500.0,
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_synth() -> FakeSynthesizer:
    return FakeSynthesizer()


@pytest.fixture
def encoder() -> AudioEncoder:
    return AudioEncoder()


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@pytest.fixture
def memory_cache() -> InMemoryPhraseCache:
    return InMemoryPhraseCache(max_entries=32, max_bytes=16 * 1024 * 1024)


@pytest.fixture
def pcm() -> np.ndarray:
    return make_pcm()


@pytest.fixture
def voice() -> VoiceProfile:
    return make_voice()
