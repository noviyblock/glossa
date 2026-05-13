from uuid import UUID

from pydantic import Field

from .base import GlossaModel


class RAGDocument(GlossaModel):
    id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class NLPRequest(GlossaModel):
    session_id: UUID
    gloss_sequence: str | None = None  # RSL→text: glosses as input
    text: str | None = None            # text→RSL: raw text as input
    domain: str = "general"
    context_hint: str | None = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    use_rag: bool = True


class NLPResult(GlossaModel):
    session_id: UUID
    text: str                  # natural language output
    gloss_sequence: str | None = None  # RSL gloss output (text→RSL mode)
    tokens_generated: int
    rag_documents: list[RAGDocument] = Field(default_factory=list)
    model_name: str
    processing_time_ms: float
    finish_reason: str = "stop"


class NLPStreamChunk(GlossaModel):
    session_id: UUID
    chunk_index: int
    delta: str
    is_final: bool = False
