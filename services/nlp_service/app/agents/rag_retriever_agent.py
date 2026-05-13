"""Agent 6 — RAG retriever agent.

Wraps the Qdrant retriever in the agent interface.
Handles query construction from state, token-budget-aware context truncation.
"""

from __future__ import annotations

from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import AgentResult, AgentStatus, TokenBudget
from .base import BaseAgent

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 1200   # hard cap before token budget check


class RAGRetrieverAgent(BaseAgent):
    """Retrieves and formats relevant documents from the vector store."""

    @property
    def name(self) -> str:
        return "rag_retriever"

    def __init__(self, llm, retriever, prompt_version: str = "1.0") -> None:
        super().__init__(llm, prompt_version)
        self._retriever = retriever

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        query = self._build_query(state)
        if not query:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SKIPPED,
                output="",
                confidence=1.0,
                metadata={"reason": "empty_query"},
            )

        domain: str = state.get("domain", "general")

        # Retrieve documents
        try:
            docs = await self._retriever.retrieve(query, domain=domain)
        except Exception as exc:
            logger.exception("rag_retriever_failed")
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FALLBACK,
                output="",
                confidence=0.0,
                error=str(exc),
            )

        if not docs:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                output="",
                confidence=1.0,
                metadata={"n_docs": 0},
            )

        context = self._format_context(docs, state)
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=context,
            confidence=float(docs[0].score) if docs else 0.0,
            metadata={
                "n_docs": len(docs),
                "top_score": round(float(docs[0].score), 3) if docs else 0.0,
                "context_chars": len(context),
            },
        )

    def _build_query(self, state: dict[str, Any]) -> str:
        gloss = state.get("gloss_sequence", "")
        text = state.get("input_text", "")
        return (gloss or text).strip()[:500]

    def _format_context(self, docs: list, state: dict[str, Any]) -> str:
        budget: TokenBudget | None = state.get("token_budget")
        parts: list[str] = []
        total_chars = 0

        for doc in docs:
            gloss = doc.metadata.get("gloss", "")
            translation = doc.metadata.get("translation", doc.text)
            line = f"• {gloss}: {translation}" if gloss else f"• {doc.text}"

            if budget and not budget.can_include_rag(total_chars + len(line)):
                break
            if total_chars + len(line) > _MAX_CONTEXT_CHARS:
                break

            parts.append(line)
            total_chars += len(line)

        return "\n".join(parts)
