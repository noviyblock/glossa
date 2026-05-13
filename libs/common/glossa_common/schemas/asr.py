from uuid import UUID

from pydantic import Field

from .base import GlossaModel


class ASRRequest(GlossaModel):
    session_id: UUID
    audio_bytes: bytes
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: str = "ru"
    beam_size: int = Field(default=5, ge=1, le=20)
    vad_filter: bool = True
    word_timestamps: bool = False


class WordTimestamp(GlossaModel):
    word: str
    start: float
    end: float
    probability: float = Field(ge=0.0, le=1.0)


class ASRStreamChunk(GlossaModel):
    session_id: UUID
    chunk_index: int = Field(ge=0)
    text: str
    is_final: bool = False
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class ASRResult(GlossaModel):
    session_id: UUID
    text: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    duration_seconds: float
    words: list[WordTimestamp] = Field(default_factory=list)
    processing_time_ms: float
