"""Agent 2 — Gloss-to-text translator.

Converts RSL gloss sequences into natural Russian text.
Incorporates RAG context and conversation history.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..domain.entities import AgentResult, AgentStatus
from ..domain.prompts import DOMAIN_HINTS, get_prompt
from .base import BaseAgent


class GlossTranslatorAgent(BaseAgent):
    """RSL gloss → natural Russian text."""

    FALLBACK_OUTPUT = "[Перевод временно недоступен]"

    @property
    def name(self) -> str:
        return "gloss_translator"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        gloss_sequence: str = state.get("gloss_sequence") or state.get("input_text", "")
        domain: str = state.get("domain", "general")
        rag_context: str = state.get("rag_context", "")
        history: str = state.get("history_text", "")
        budget: int = state.get("generation_budget", 256)

        domain_hint = DOMAIN_HINTS.get(domain, "")
        rag_block = (
            f"Глоссарий РЖЯ (релевантные примеры):\n{rag_context}\n\n"
            if rag_context else ""
        )
        history_block = history or "Нет истории"

        prompt = get_prompt("gloss_translator", self._prompt_version)
        user_content = prompt.template.format_map({
            "gloss_sequence": gloss_sequence,
            "domain": domain,
            "domain_hint": domain_hint,
            "rag_context": rag_block,
            "history": history_block,
        })

        messages = [{"role": "user", "content": user_content}]
        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=min(budget, 512),
            temperature=0.1,
            stop=None,
        )

        confidence = self._estimate_confidence(text, gloss_sequence)
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=text.strip(),
            confidence=confidence,
            tokens_used=tokens,
            metadata={"domain": domain, "gloss_len": len(gloss_sequence.split())},
        )

    async def stream(self, state: dict[str, Any]) -> AsyncIterator[str]:
        gloss_sequence: str = state.get("gloss_sequence") or state.get("input_text", "")
        domain: str = state.get("domain", "general")
        rag_context: str = state.get("rag_context", "")
        history: str = state.get("history_text", "")
        budget: int = state.get("generation_budget", 256)

        domain_hint = DOMAIN_HINTS.get(domain, "")
        rag_block = f"Глоссарий:\n{rag_context}\n\n" if rag_context else ""

        prompt = get_prompt("gloss_translator", self._prompt_version)
        user_content = prompt.template.format_map({
            "gloss_sequence": gloss_sequence,
            "domain": domain,
            "domain_hint": domain_hint,
            "rag_context": rag_block,
            "history": history or "Нет истории",
        })

        messages = [{"role": "user", "content": user_content}]
        async for token in self._llm.chat_stream(
            messages=messages,
            max_tokens=min(budget, 512),
            temperature=0.1,
        ):
            yield token

    def _estimate_confidence(self, output: str, gloss_input: str) -> float:
        """Heuristic confidence: penalize empty / very short / repetitive output."""
        if not output.strip():
            return 0.0
        words = output.split()
        gloss_words = gloss_input.upper().split()
        # If output still looks like raw glosses → low confidence
        caps_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)
        if caps_ratio > 0.5:
            return 0.3
        # Length ratio: expect ~1.5-3× expansion from glosses
        length_ratio = len(words) / max(len(gloss_words), 1)
        if length_ratio < 0.5:
            return 0.5
        return min(0.95, 0.7 + 0.05 * min(len(words), 5))
