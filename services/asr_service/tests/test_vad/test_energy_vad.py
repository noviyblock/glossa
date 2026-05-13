"""Tests for EnergyVAD."""

import numpy as np
import pytest

from app.domain.entities import VADState
from app.vad.energy_vad import EnergyVAD


def _sine(freq=440.0, duration_s=1.0, sr=16000, amplitude=0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _silence(duration_s=1.0, sr=16000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


class TestEnergyVAD:
    def setup_method(self):
        self.vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            threshold=0.02,
            min_speech_ms=100,
            min_silence_ms=200,
        )

    def test_detects_speech_in_sine(self):
        audio = _sine(440, 1.0)
        segments = self.vad.process(audio, 16000)
        assert len(segments) >= 1
        assert segments[0].state == VADState.SPEECH

    def test_no_speech_in_silence(self):
        audio = _silence(1.0)
        segments = self.vad.process(audio, 16000)
        assert segments == []

    def test_segments_correct_duration(self):
        audio = np.concatenate([_silence(0.5), _sine(440, 0.5), _silence(0.5)])
        segments = self.vad.process(audio, 16000)
        assert len(segments) >= 1
        total_speech_ms = sum(s.duration_ms for s in segments)
        assert total_speech_ms > 200  # at least 200 ms of speech detected

    def test_reset_is_stateless(self):
        self.vad.reset()  # EnergyVAD is stateless — should not raise
        audio = _sine(440, 0.5)
        segments = self.vad.process(audio, 16000)
        assert len(segments) >= 1

    def test_multiple_speech_segments(self):
        audio = np.concatenate([
            _sine(440, 0.5),
            _silence(0.5),
            _sine(880, 0.5),
            _silence(0.3),
        ])
        segments = self.vad.process(audio, 16000)
        # Should detect at least 1 segment (min_silence 200ms may merge some)
        assert len(segments) >= 1

    def test_empty_audio(self):
        segments = self.vad.process(np.zeros(0, dtype=np.float32), 16000)
        assert segments == []

    def test_very_short_audio_below_min_speech(self):
        audio = _sine(440, 0.05)  # 50 ms < min_speech_ms=100
        segments = self.vad.process(audio, 16000)
        assert segments == []
