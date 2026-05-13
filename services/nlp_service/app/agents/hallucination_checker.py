"""Hallucination mitigation agent.

Cross-checks generated output against RAG context.
Returns a list of potential inconsistencies.
"""

from __future__ import annotations

import json
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import AgentResult, AgentStatus
from ..domain.prompts import get_prompt
from .base import BaseAgent

logger = get_logger(__name__)


class HallucinationCheckerAgent(BaseAgent):
    """Post-generation factual consistency check."""

    FALLBACK_OUTPUT = "[]"

    @property
    def name(self) -> str:
        return "hallucination_checker"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        output_text: str = state.get("output_text", "")
        rag_context: str = state.get("rag_context", "")

        # Skip if no RAG context — can't verify without a source of truth
        if not rag_context or not output_text:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SKIPPED,
                output="[]",
                confidence=1.0,
            )

        prompt = get_prompt("hallucination_checker", self._prompt_version)
        user_content = prompt.template.format_map({
            "rag_context": rag_context[:800],
            "output_text": output_text[:600],
        })

        messages = [{"role": "user", "content": user_content}]
        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=256,
            temperature=0.0,
            stop=None,
        )

        # Parse flags JSON
        flags: list[str] = []
        try:
            data = json.loads(text.strip())
            flags = data.get("flags", [])
        except (json.JSONDecodeError, AttributeError):
            # LLM didn't return valid JSON — extract inline
            flags = []

        if flags:
            logger.warning(
                "hallucination_flags_detected",
                n=len(flags),
                flags=flags[:3],
            )

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=json.dumps(flags, ensure_ascii=False),
            confidence=1.0 - min(0.5, len(flags) * 0.15),
            tokens_used=tokens,
            metadata={"n_flags": len(flags)},
        )
