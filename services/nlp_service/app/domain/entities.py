"""NLP domain entities — pure Python, no I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


# ── Enumerations ──────────────────────────────────────────────────────────────

class Intent(StrEnum):
    GLOSS_TO_TEXT   = "gloss_to_text"
    TEXT_TO_GLOSS   = "text_to_gloss"
    SIMPLIFY        = "simplify"
    QUESTION        = "question"
    CLARIFY         = "clarify"
    CHITCHAT        = "chitchat"
    UNKNOWN         = "unknown"


class Domain(StrEnum):
    GENERAL  = "general"
    MEDICAL  = "medical"
    BANKING  = "banking"
    LEGAL    = "legal"
    EDUCATION = "education"


class SafetyLabel(StrEnum):
    SAFE     = "safe"
    WARN     = "warn"       # borderline — pass with warning
    BLOCK    = "block"      # reject entirely


class AgentStatus(StrEnum):
    SUCCESS  = "success"
    FALLBACK = "fallback"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ── Prompt versioning ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    template: str
    language: str = "ru"
    created_at: float = field(default_factory=time.time)


# ── Per-agent result ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent: str
    status: AgentStatus
    output: str = ""
    confidence: float = 1.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.SUCCESS


# ── Conversation memory ───────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str           # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: Intent | None = None
    domain: Domain | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemory:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    dominant_domain: Domain = Domain.GENERAL
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_turn(self, role: str, content: str, **kwargs) -> None:
        self.turns.append(ConversationTurn(role=role, content=content, **kwargs))
        self.updated_at = time.time()

    def recent_turns(self, n: int = 6) -> list[ConversationTurn]:
        return self.turns[-n:]

    def to_messages(self, n: int = 6) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.recent_turns(n)]


# ── Token budget ──────────────────────────────────────────────────────────────

@dataclass
class TokenBudget:
    total: int = 2048
    system_reserved: int = 256
    rag_reserved: int = 512
    history_reserved: int = 256
    used_prompt: int = 0
    used_generation: int = 0

    @property
    def generation_budget(self) -> int:
        used = self.system_reserved + self.rag_reserved + self.history_reserved + self.used_prompt
        return max(64, self.total - used)

    def can_include_rag(self, rag_tokens: int) -> bool:
        return rag_tokens <= self.rag_reserved


# ── Graph-level request/result ────────────────────────────────────────────────

@dataclass
class NLPRequest:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    input_text: str = ""
    gloss_sequence: str | None = None
    language: str = "ru"
    domain: Domain | None = None
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.1
    submitted_at: float = field(default_factory=time.perf_counter)


@dataclass
class NLPResponse:
    request_id: str
    session_id: str
    output_text: str
    intent: Intent
    domain: Domain
    confidence: float
    safety: SafetyLabel
    agent_trace: list[AgentResult] = field(default_factory=list)
    rag_docs_used: int = 0
    tokens_used: int = 0
    latency_ms: float = 0.0
    is_fallback: bool = False
    hallucination_flags: list[str] = field(default_factory=list)
