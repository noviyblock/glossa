"""Tests for AudioEncoder — WAV encoding and framing."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from app.audio.encoder import (
    AudioEncoder,
    FrameFlags,
    FrameType,
    _FRAME_HEADER_SIZE,
    pack_frame,
    unpack_frame_header,
)
from app.domain.entities import AudioFormat


@pytest.fixture
def enc() -> AudioEncoder:
    return AudioEncoder()


# ── pcm_to_wav ────────────────────────────────────────────────────────────────

def test_pcm_to_wav_produces_valid_wav(enc):
    pcm = np.zeros(24000, dtype=np.float32)  # 1s silent
    wav = enc.pcm_to_wav(pcm, sample_rate=24000)

    buf = io.BytesIO(wav)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 24000


def test_pcm_to_wav_clips_values(enc):
    pcm = np.array([1.5, -1.5, 0.5], dtype=np.float32)
    wav = enc.pcm_to_wav(pcm, 24000)
    # Should not raise; clipping applied internally
    assert len(wav) > 0


def test_pcm_to_wav_non_default_sample_rate(enc):
    pcm = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
    wav = enc.pcm_to_wav(pcm, sample_rate=16000)
    buf = io.BytesIO(wav)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == 16000


# ── audio_duration ────────────────────────────────────────────────────────────

def test_audio_duration(enc):
    pcm = np.zeros(24000, dtype=np.float32)
    assert enc.audio_duration(pcm, 24000) == pytest.approx(1.0)


def test_audio_duration_zero_rate(enc):
    assert enc.audio_duration(np.zeros(100), 0) == 0.0


# ── encode dispatch ───────────────────────────────────────────────────────────

def test_encode_wav_dispatches_correctly(enc):
    pcm = np.zeros(1000, dtype=np.float32)
    result = enc.encode(pcm, 24000, AudioFormat.WAV)
    assert result[:4] == b"RIFF"  # WAV magic


def test_encode_ogg_falls_back_to_wav_when_unavailable(enc, monkeypatch):
    # Simulate soundfile not installed
    import app.audio.encoder as enc_module
    monkeypatch.setattr(enc_module, "_OGG_AVAILABLE", False)
    pcm = np.zeros(1000, dtype=np.float32)
    result = enc.encode(pcm, 24000, AudioFormat.OGG)
    assert result[:4] == b"RIFF"  # fell back to WAV


# ── pcm_to_raw_s16le ─────────────────────────────────────────────────────────

def test_raw_s16le_correct_length(enc):
    pcm = np.zeros(100, dtype=np.float32)
    raw = enc.pcm_to_raw_s16le(pcm)
    assert len(raw) == 100 * 2  # 2 bytes per sample


def test_raw_s16le_full_scale_positive(enc):
    pcm = np.array([1.0], dtype=np.float32)
    raw = enc.pcm_to_raw_s16le(pcm)
    val = struct.unpack("<h", raw)[0]
    assert val == 32767


# ── Frame packing ─────────────────────────────────────────────────────────────

def test_pack_frame_header_size():
    frame = pack_frame(b"hello", FrameType.AUDIO, FrameFlags.NONE, seq=0)
    assert len(frame) == _FRAME_HEADER_SIZE + 5


def test_unpack_frame_header_round_trip():
    payload = b"test payload"
    frame = pack_frame(payload, FrameType.FINAL, FrameFlags.LAST_CHUNK, seq=42)
    ftype, flags, seq, length = unpack_frame_header(frame)
    assert ftype == FrameType.FINAL
    assert flags & FrameFlags.LAST_CHUNK
    assert seq == 42
    assert length == len(payload)


def test_unpack_frame_header_too_short():
    with pytest.raises(ValueError, match="too short"):
        unpack_frame_header(b"\x01\x02")


def test_pack_frame_seq_wraps_at_16bit():
    frame = pack_frame(b"x", seq=0xFFFF + 1)
    _, _, seq, _ = unpack_frame_header(frame)
    assert seq == 0  # wrapped
