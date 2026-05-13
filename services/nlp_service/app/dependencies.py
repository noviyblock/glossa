"""FastAPI dependency resolvers."""

from __future__ import annotations

from fastapi import Request

from .graphs.orchestrator import NLPOrchestrator
from .memory.conversation import InMemoryConversationStore


def get_orchestrator(request: Request) -> NLPOrchestrator:
    return request.app.state.orchestrator


def get_memory_store(request: Request) -> InMemoryConversationStore:
    return request.app.state.memory_store
