"""Pydantic v2 request schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glossa_common.schemas.base import GlossaModel


class TranscribeRequest(GlossaModel):
    """Single-shot transcription from raw audio bytes (base64-encoded)."""

    audio_b64: str = Field(..., description="Base64-encoded PCM int16 or WAV audio")
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: str | None = Field(default=None, description="BCP-47 code or null for auto-detect")
    beam_size: int = Field(default=5, ge=1, le=20)
    word_timestamps: bool = False
    medical_adaptation: bool = True


class TranscribeChunkRequest(GlossaModel):
    """Streaming chunk sent over HTTP (fallback when WebSocket unavailable)."""

    session_id: str
    chunk_b64: str
    seq: int = 0
    sample_rate: int = 16000
    is_final: bool = False


class BenchmarkRequest(GlossaModel):
    audio_duration_s: float = Field(default=5.0, ge=0.5, le=30.0)
    n_samples: int = Field(default=20, ge=5, le=200)
    language: str = "ru"
    beam_size: int = Field(default=5, ge=1, le=20)
    warmup_iters: int = Field(default=3, ge=1, le=20)


class AddMedicalTermRequest(GlossaModel):
    term: str = Field(..., min_length=1)
    correction: str = Field(..., min_length=1)
    language: str = "ru"


class DetectLanguageRequest(GlossaModel):
    audio_b64: str
    sample_rate: int = 16000
