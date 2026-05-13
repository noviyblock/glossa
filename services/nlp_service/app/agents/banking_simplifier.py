"""Agent 4 — Banking simplifier.

Translates complex financial/banking text into plain language.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..domain.entities import AgentResult, AgentStatus
from ..domain.prompts import get_prompt
from .base import BaseAgent


class BankingSimplifierAgent(BaseAgent):
    """Simplifies banking and financial terminology."""

    FALLBACK_OUTPUT = "[Банковский упрощитель временно недоступен]"

    @property
    def name(self) -> str:
        return "banking_simplifier"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        input_text: str = state.get("input_text", "") or state.get("output_text", "")
        rag_context: str = state.get("rag_context", "")
        budget: int = state.get("generation_budget", 512)

        rag_block = f"Финансовый глоссарий:\n{rag_context}\n\n" if rag_context else ""

        prompt = get_prompt("banking_simplifier", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "rag_context": rag_block,
        })

        messages = [{"role": "user", "content": user_content}]
        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=min(budget, 768),
            temperature=0.1,
            stop=None,
        )

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=text.strip(),
            confidence=0.87,
            tokens_used=tokens,
            metadata={"domain": "banking"},
        )

    async def stream(self, state: dict[str, Any]) -> AsyncIterator[str]:
        input_text: str = state.get("input_text", "") or state.get("output_text", "")
        rag_context: str = state.get("rag_context", "")
        budget: int = state.get("generation_budget", 512)

        rag_block = f"Финансовый глоссарий:\n{rag_context}\n\n" if rag_context else ""
        prompt = get_prompt("banking_simplifier", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "rag_context": rag_block,
        })

        async for token in self._llm.chat_stream(
            messages=[{"role": "user", "content": user_content}],
            max_tokens=min(budget, 768),
            temperature=0.1,
        ):
            yield token
