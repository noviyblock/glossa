from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field

from .base import GlossaModel


class TranslationMode(StrEnum):
    RSL_TO_TEXT = "rsl_to_text"
    TEXT_TO_RSL = "text_to_rsl"


class TranslationRequest(GlossaModel):
    session_id: UUID = Field(default_factory=uuid4)
    mode: TranslationMode
    source_language: str = "ru"
    target_language: str = "ru"
    context_hint: str | None = Field(default=None, max_length=500)
    domain: str = Field(default="general", pattern=r"^(general|medical|banking|legal)$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TranslationChunk(GlossaModel):
    session_id: UUID
    chunk_index: int = Field(ge=0)
    text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    is_final: bool = False
    stage: str | None = None  # cv | asr | nlp | tts
    latency_ms: float | None = None


class TranslationResult(GlossaModel):
    session_id: UUID
    mode: TranslationMode
    text: str
    audio_bytes: bytes | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    total_latency_ms: float
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    tokens_used: int | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebSocketMessage(GlossaModel):
    type: str  # "chunk" | "result" | "error" | "ping"
    session_id: UUID
    payload: dict[str, object] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
