"""Agent 3 — Medical simplifier.

Transforms complex medical text into plain language for deaf/hard-of-hearing patients.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..domain.entities import AgentResult, AgentStatus
from ..domain.prompts import get_prompt
from .base import BaseAgent


class MedicalSimplifierAgent(BaseAgent):
    """Simplifies medical terminology for RSL users."""

    FALLBACK_OUTPUT = "[Медицинский упрощитель временно недоступен]"

    @property
    def name(self) -> str:
        return "medical_simplifier"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        input_text: str = state.get("input_text", "") or state.get("output_text", "")
        rag_context: str = state.get("rag_context", "")
        budget: int = state.get("generation_budget", 512)

        rag_block = (
            f"Медицинский глоссарий:\n{rag_context}\n\n" if rag_context else ""
        )

        prompt = get_prompt("medical_simplifier", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "rag_context": rag_block,
        })

        messages = [{"role": "user", "content": user_content}]
        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=min(budget, 768),
            temperature=0.05,  # low temperature for factual accuracy
            stop=None,
        )

        confidence = self._assess_simplification(text, input_text)
        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=text.strip(),
            confidence=confidence,
            tokens_used=tokens,
            metadata={"domain": "medical"},
        )

    async def stream(self, state: dict[str, Any]) -> AsyncIterator[str]:
        input_text: str = state.get("input_text", "") or state.get("output_text", "")
        rag_context: str = state.get("rag_context", "")
        budget: int = state.get("generation_budget", 512)

        rag_block = f"Медицинский глоссарий:\n{rag_context}\n\n" if rag_context else ""
        prompt = get_prompt("medical_simplifier", self._prompt_version)
        user_content = prompt.template.format_map({
            "input_text": input_text,
            "rag_context": rag_block,
        })

        async for token in self._llm.chat_stream(
            messages=[{"role": "user", "content": user_content}],
            max_tokens=min(budget, 768),
            temperature=0.05,
        ):
            yield token

    def _assess_simplification(self, output: str, input_text: str) -> float:
        if not output.strip():
            return 0.0
        out_words = output.split()
        in_words = input_text.split()
        # Simplified text should be at least as long as input (not truncated)
        if len(out_words) < len(in_words) * 0.3:
            return 0.4
        # Check that output doesn't just copy the input verbatim
        overlap = len(set(out_words) & set(in_words)) / max(len(set(out_words)), 1)
        if overlap > 0.9:
            return 0.5  # likely no simplification occurred
        return 0.88
