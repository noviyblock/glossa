"""Pydantic v2 response schemas."""

from __future__ import annotations

from glossa_common.schemas.base import GlossaModel


class AgentTraceItem(GlossaModel):
    agent: str
    status: str
    latency_ms: float
    confidence: float
    tokens_used: int
    error: str | None = None


class NLPChatResponse(GlossaModel):
    request_id: str
    session_id: str
    output_text: str
    intent: str
    domain: str
    confidence: float
    safety: str
    rag_docs_used: int
    tokens_used: int
    latency_ms: float
    is_fallback: bool
    hallucination_flags: list[str] = []
    agent_trace: list[AgentTraceItem] = []


class TranslateGlossResponse(GlossaModel):
    session_id: str
    gloss_sequence: str
    translated_text: str
    domain: str
    confidence: float
    rag_docs_used: int
    inference_ms: float
    is_fallback: bool


class SimplifyResponse(GlossaModel):
    session_id: str
    original_text: str
    simplified_text: str
    domain: str
    confidence: float
    inference_ms: float


class PromptListResponse(GlossaModel):
    prompts: list[dict]


class ConversationStatsResponse(GlossaModel):
    session_id: str
    turn_count: int
    dominant_domain: str
    has_summary: bool


class MessageResponse(GlossaModel):
    message: str
