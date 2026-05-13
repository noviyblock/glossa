"""Pydantic v2 response schemas."""

from __future__ import annotations

from glossa_common.schemas.base import GlossaModel


class WordTimestampResponse(GlossaModel):
    word: str
    start: float
    end: float
    probability: float


class SegmentResponse(GlossaModel):
    text: str
    start: float
    end: float
    confidence: float
    no_speech_prob: float


class TranscribeResponse(GlossaModel):
    text: str
    corrected_text: str
    language: str
    language_probability: float
    duration_s: float
    inference_ms: float
    segments: list[SegmentResponse] = []
    words: list[WordTimestampResponse] = []
    medical_terms_found: list[str] = []


class PartialTranscriptResponse(GlossaModel):
    type: str = "partial"
    text: str
    language: str
    chunk_seq: int


class FinalTranscriptResponse(GlossaModel):
    type: str = "final"
    text: str
    corrected_text: str
    language: str
    language_probability: float
    segments: list[SegmentResponse] = []
    words: list[WordTimestampResponse] = []
    inference_ms: float
    total_latency_ms: float
    medical_terms_found: list[str] = []


class BenchmarkResponse(GlossaModel):
    model_size: str
    device: str
    compute_type: str
    n_samples: int
    audio_duration_s: float
    mean_inference_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rtf: float


class DetectLanguageResponse(GlossaModel):
    language: str
    probability: float


class SessionInfoResponse(GlossaModel):
    session_id: str
    reconnect_token: str
    language: str
    status: str
    chunk_count: int
    transcript_count: int
    total_audio_s: float


class MessageResponse(GlossaModel):
    message: str
