"""Agent 1 — Intent classifier.

Classifies the user's input into one of the Intent enum values.
Uses few-shot prompting for robustness, falls back to UNKNOWN.
"""

from __future__ import annotations

import re
from typing import Any

from ..domain.entities import AgentResult, AgentStatus, Intent
from ..domain.prompts import get_prompt
from .base import BaseAgent

_VALID_INTENTS = {i.value for i in Intent}

# Heuristic pre-filter: if input is all-caps words separated by spaces → likely glosses
_GLOSS_PATTERN = re.compile(r"^[А-ЯA-Z][А-ЯA-Z\s\-]{3,}$")


class IntentClassifierAgent(BaseAgent):
    """Classifies intent using LLM + heuristic pre-check."""

    FALLBACK_OUTPUT = Intent.UNKNOWN.value

    @property
    def name(self) -> str:
        return "intent_classifier"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        input_text: str = state.get("input_text", "")
        gloss_sequence: str | None = state.get("gloss_sequence")

        # Fast heuristic: explicit gloss input
        if gloss_sequence:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                output=Intent.GLOSS_TO_TEXT.value,
                confidence=0.99,
                metadata={"method": "heuristic_gloss"},
            )

        # Heuristic: all-caps word sequence → treat as glosses
        if _GLOSS_PATTERN.match(input_text.strip()):
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                output=Intent.GLOSS_TO_TEXT.value,
                confidence=0.85,
                metadata={"method": "heuristic_caps"},
            )

        # LLM classification
        prompt = get_prompt("intent_classifier", self._prompt_version)
        user_msg = prompt.template.format_map({"input_text": input_text})
        messages = [{"role": "user", "content": user_msg}]

        text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=16,
            temperature=0.0,
            stop=["\n", " "],
        )

        intent_raw = text.strip().lower().split()[0] if text.strip() else ""
        intent = intent_raw if intent_raw in _VALID_INTENTS else Intent.UNKNOWN.value

        # Confidence: full match = high, partial match = medium, no match = low
        confidence = 0.95 if intent != Intent.UNKNOWN.value else 0.30

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=intent,
            confidence=confidence,
            tokens_used=tokens,
            metadata={"raw_response": text, "method": "llm"},
        )
