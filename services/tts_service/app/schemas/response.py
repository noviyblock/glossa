"""Pydantic v2 response schemas for TTS endpoints."""

from __future__ import annotations

from glossa_common.schemas.base import GlossaModel

from ..domain.entities import AudioFormat


class SynthesizeResponse(GlossaModel):
    """Full audio response — base64-encoded bytes in JSON."""

    session_id: str
    request_id: str
    audio_b64: str               # base64-encoded audio
    format: AudioFormat
    sample_rate: int
    duration_s: float
    char_count: int
    voice_id: str
    from_cache: bool
    inference_ms: float
    total_ms: float


class VoiceProfileResponse(GlossaModel):
    id: str
    display_name: str
    language: str
    gender: str
    engine: str
    sample_rate: int


class VoiceListResponse(GlossaModel):
    voices: list[VoiceProfileResponse]


class CacheStatsResponse(GlossaModel):
    l1: dict
    l2: dict


class BenchmarkResponse(GlossaModel):
    n_runs: int
    text_len: int
    voice_id: str
    sample_rate: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    rtf: float
    chars_per_sec: float
    format: AudioFormat


class PreloadResponse(GlossaModel):
    loaded: int
    total: int
    voice_id: str
