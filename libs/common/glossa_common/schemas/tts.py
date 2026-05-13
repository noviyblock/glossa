from uuid import UUID

from pydantic import Field

from .base import GlossaModel


class TTSRequest(GlossaModel):
    session_id: UUID
    text: str = Field(min_length=1, max_length=5000)
    speaker: str = "aidar"
    language: str = "ru"
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    put_accent: bool = True
    put_yo: bool = True


class TTSResult(GlossaModel):
    session_id: UUID
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"
    processing_time_ms: float


class TTSStreamChunk(GlossaModel):
    session_id: UUID
    chunk_index: int = Field(ge=0)
    audio_bytes: bytes
    is_final: bool = False
