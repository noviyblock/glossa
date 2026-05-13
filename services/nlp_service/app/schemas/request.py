"""Pydantic v2 request schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glossa_common.schemas.base import GlossaModel

from ..domain.entities import Domain


class NLPChatRequest(GlossaModel):
    """Main chat/translation request."""

    session_id: str = Field(default="", description="Session ID for conversation memory")
    input_text: str = Field(..., min_length=1, max_length=4096)
    gloss_sequence: str | None = Field(default=None, description="RSL gloss sequence (overrides input_text for translation)")
    language: str = Field(default="ru", description="BCP-47 language code")
    domain: Domain = Domain.GENERAL
    stream: bool = False
    max_tokens: int = Field(default=512, ge=32, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class TranslateGlossRequest(GlossaModel):
    """Dedicated gloss-to-text translation endpoint."""

    session_id: str = ""
    gloss_sequence: str = Field(..., min_length=1)
    domain: Domain = Domain.GENERAL
    stream: bool = False
    max_tokens: int = Field(default=256, ge=32, le=1024)


class SimplifyRequest(GlossaModel):
    """Text simplification for medical or banking domain."""

    session_id: str = ""
    text: str = Field(..., min_length=1, max_length=4096)
    domain: Literal["medical", "banking"] = "medical"
    stream: bool = False
    max_tokens: int = Field(default=512, ge=64, le=2048)


class AddPromptRequest(GlossaModel):
    """Register a new prompt version (admin endpoint)."""

    agent: str
    version: str
    template: str
    language: str = "ru"


class ConversationResetRequest(GlossaModel):
    session_id: str
