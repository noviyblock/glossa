"""Abstract ports for NLP domain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from .entities import AgentResult, ConversationMemory, NLPRequest, NLPResponse


class AgentPort(ABC):
    """Base interface for all graph agents."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> AgentResult:
        """Execute the agent and return a structured result."""

    async def stream(self, state: dict[str, Any]) -> AsyncIterator[str]:
        """Optional: yield tokens as they are generated."""
        result = await self.run(state)
        yield result.output


class LLMPort(ABC):
    """Abstract LLM inference backend."""

    @abstractmethod
    async def ensure_loaded(self) -> None: ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[str, int]:
        """Returns (text, tokens_used)."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield tokens as they are generated."""


class EmbedderPort(ABC):
    """Text embedding interface."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class MemoryStorePort(ABC):
    """Conversation memory persistence."""

    @abstractmethod
    async def load(self, session_id: str) -> ConversationMemory | None: ...

    @abstractmethod
    async def save(self, memory: ConversationMemory) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...
