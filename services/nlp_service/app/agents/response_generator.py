"""General response generator — handles QUESTION / CHITCHAT / CLARIFY intents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..domain.entities import AgentResult, AgentStatus
from ..domain.prompts import DOMAIN_HINTS, get_prompt
from .base import BaseAgent


class ResponseGeneratorAgent(BaseAgent):
    """Generates open-domain responses for non-translation intents."""

    FALLBACK_OUTPUT = "[Ответ временно недоступен]"

    @property
    def name(self) -> str:
        return "response_generator"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        input_text: str = state.get("input_text", "")
        domain: str = state.get("domain", "general")
        rag_context: str = state.get("rag_context", "")
        history: str = state.get("history_text", "")
        budget: int = state.get("generation_budget", 384)

        domain_hint = DOMAIN_HINTS.get(domain, "")
        rag_block = f"Контекст из базы знаний:\n{rag_context}\n\n" if rag_context else ""

        prompt = get_prompt("response_generator", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "domain": domain,
            "domain_hint": domain_hint,
            "rag_context": rag_block,
            "history": history or "Нет истории",
        })

        messages = [{"role": "user", "content": user_content}]
        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=min(budget, 512),
            temperature=0.3,
            stop=None,
        )

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=text.strip(),
            confidence=0.80,
            tokens_used=tokens,
            metadata={"domain": domain},
        )

    async def stream(self, state: dict[str, Any]) -> AsyncIterator[str]:
        input_text: str = state.get("input_text", "")
        domain: str = state.get("domain", "general")
        rag_context: str = state.get("rag_context", "")
        history: str = state.get("history_text", "")
        budget: int = state.get("generation_budget", 384)

        domain_hint = DOMAIN_HINTS.get(domain, "")
        rag_block = f"Контекст:\n{rag_context}\n\n" if rag_context else ""

        prompt = get_prompt("response_generator", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "domain": domain,
            "domain_hint": domain_hint,
            "rag_context": rag_block,
            "history": history or "Нет истории",
        })

        async for token in self._llm.chat_stream(
            messages=[{"role": "user", "content": user_content}],
            max_tokens=min(budget, 512),
            temperature=0.3,
        ):
            yield token
