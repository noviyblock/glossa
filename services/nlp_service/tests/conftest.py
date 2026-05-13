"""Shared pytest fixtures for the NLP service test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.entities import AgentResult, AgentStatus, Intent, SafetyLabel


# ── Fake LLM ─────────────────────────────────────────────────────────────────

class FakeLLM:
    """Controllable LLM stub — callers set .response before each call."""

    def __init__(self, response: str = "safe", tokens: int = 4) -> None:
        self.response = response
        self.tokens = tokens
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 16,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response, self.tokens

    async def chat_stream(self, messages, max_tokens=512, temperature=0.1):
        for token in self.response.split():
            yield token


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# ── Fake RAG retriever ────────────────────────────────────────────────────────

class FakeRetriever:
    def __init__(self, docs=None):
        self._docs = docs or []

    async def retrieve(self, query: str, domain: str | None = None):
        return self._docs

    def format_context(self, docs) -> str:
        return "\n".join(d.text for d in docs)


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


# ── Fake memory store ─────────────────────────────────────────────────────────

class FakeMemoryStore:
    def __init__(self):
        self._data: dict = {}

    async def get(self, session_id: str):
        return self._data.get(session_id)

    async def save(self, memory) -> None:
        self._data[memory.session_id] = memory

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


@pytest.fixture
def fake_memory_store() -> FakeMemoryStore:
    return FakeMemoryStore()


# ── Common graph state factories ──────────────────────────────────────────────

def make_state(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "request_id": "test-req-001",
        "session_id": "test-session-001",
        "input_text": "Как взять ипотеку?",
        "gloss_sequence": None,
        "language": "ru",
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.1,
        "domain": "general",
        "intent": Intent.UNKNOWN.value,
        "intent_confidence": 0.0,
        "history_text": "",
        "turn_count": 0,
        "generation_budget": 256,
        "rag_context": "",
        "rag_docs_used": 0,
        "safety_label": SafetyLabel.SAFE.value,
        "safety_check_text": "",
        "input_safety_ok": True,
        "output_text": "",
        "tokens_used": 0,
        "generation_confidence": 0.0,
        "hallucination_flags": [],
        "final_confidence": 0.0,
        "agent_trace": [],
        "error": None,
        "latency_ms": 0.0,
    }
    base.update(overrides)
    return base
