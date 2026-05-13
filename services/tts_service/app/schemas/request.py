"""Pydantic v2 request schemas for TTS endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glossa_common.schemas.base import GlossaModel

from ..domain.entities import AudioFormat


class SynthesizeRequest(GlossaModel):
    """Full synthesis — returns complete encoded audio."""

    session_id: str = ""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Field(default="aidar", description="Voice profile ID")
    language: str = Field(default="ru", description="BCP-47 language tag")
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    format: AudioFormat = AudioFormat.WAV
    put_accent: bool = True
    put_yo: bool = True
    priority: int = Field(default=0, ge=0, le=10, description="Queue priority (0=highest)")


class SynthesizeStreamRequest(GlossaModel):
    """Chunked streaming synthesis — sentence-by-sentence."""

    session_id: str = ""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = "aidar"
    language: str = "ru"
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    format: AudioFormat = AudioFormat.WAV
    put_accent: bool = True
    put_yo: bool = True
    chunk_size_chars: int = Field(default=150, ge=50, le=400)


class BenchmarkRequest(GlossaModel):
    """Benchmark synthesis latency."""

    text: str = Field(
        default="Добрый день, как я могу вам помочь сегодня?",
        min_length=1,
        max_length=500,
    )
    voice_id: str = "aidar"
    n_runs: int = Field(default=10, ge=1, le=100)
    format: AudioFormat = AudioFormat.WAV


class PreloadRequest(GlossaModel):
    """Preload phrases into the cache."""

    phrases: list[str] = Field(
        default=[],
        description="List of phrases to preload; empty = use built-in defaults",
    )
    voice_id: str = "aidar"
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
