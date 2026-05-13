"""Shared pytest fixtures for ASR service tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain.entities import ASRSession, TranscriptSegment, WordTimestamp


# ── Audio helpers ─────────────────────────────────────────────────────────────

def sine_wave(freq: float = 440.0, duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave as float32 PCM."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def silence(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def pcm_int16_bytes(audio: np.ndarray) -> bytes:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def audio_1s(sample_rate) -> np.ndarray:
    return sine_wave(440.0, 1.0, sample_rate)


@pytest.fixture
def audio_5s(sample_rate) -> np.ndarray:
    return sine_wave(440.0, 5.0, sample_rate)


@pytest.fixture
def silence_1s(sample_rate) -> np.ndarray:
    return silence(1.0, sample_rate)


@pytest.fixture
def fake_session() -> ASRSession:
    return ASRSession(session_id="test-session-001", language="ru", sample_rate=16000)


@pytest.fixture
def fake_segment() -> TranscriptSegment:
    return TranscriptSegment(
        text="Здравствуйте доктор",
        start=0.0,
        end=1.5,
        avg_logprob=-0.3,
        no_speech_prob=0.02,
    )


class MockTranscriber:
    """Minimal mock TranscriberPort."""
    model_size = "mock"
    device = "cpu"
    compute_type = "int8"

    def __init__(self, segments: list[TranscriptSegment] | None = None) -> None:
        self._segments = segments or [
            TranscriptSegment(text="Привет", start=0.0, end=0.5, avg_logprob=-0.2, no_speech_prob=0.01),
        ]
        self.transcribe_call_count = 0

    async def ensure_loaded(self) -> None:
        pass

    async def transcribe(self, audio, sample_rate=16000, language=None, beam_size=5):
        self.transcribe_call_count += 1
        return self._segments

    async def transcribe_streaming(self, audio, sample_rate=16000, language=None, beam_size=5):
        for seg in self._segments:
            yield seg

    async def detect_language(self, audio):
        return "ru", 0.99

    async def warmup(self, iterations=3) -> float:
        return 5.0


@pytest.fixture
def mock_transcriber():
    return MockTranscriber()
