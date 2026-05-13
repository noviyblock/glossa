"""Agent 7 — Safety filter.

Screens both the user input (pre-check) and the generated response (post-check).
Uses LLM classification + keyword blacklist for defence-in-depth.
"""

from __future__ import annotations

import re
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import AgentResult, AgentStatus, SafetyLabel
from ..domain.prompts import get_prompt
from .base import BaseAgent

logger = get_logger(__name__)

# Hard keyword blocklist — no LLM needed for these
_BLOCK_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(взорв|бомб|оружи|теракт)\w*\b",
        r"\b(убий|убит|убью)\w*\b",
        r"\b(самоуби|повеш|вскрой вены)\w*\b",
        r"\b(скам|развод\s+на\s+деньги)\w*\b",
        r"\bномер\s+карты\b",
        r"\bCVV\b",
        r"\bпин.?код\b",
    ]
]

_WARN_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(алкоголь|наркот)\w*\b",
        r"\b(суицид|депресси)\w*\b",
    ]
]


class SafetyFilterAgent(BaseAgent):
    """Pre- and post-generation safety screening."""

    FALLBACK_OUTPUT = SafetyLabel.WARN.value

    @property
    def name(self) -> str:
        return "safety_filter"

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        text: str = state.get("safety_check_text", state.get("input_text", ""))

        # Fast keyword check first (no LLM latency)
        keyword_label = self._keyword_check(text)
        if keyword_label == SafetyLabel.BLOCK:
            logger.warning("safety_keyword_block", text_preview=text[:80])
            return AgentResult(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                output=SafetyLabel.BLOCK.value,
                confidence=0.99,
                metadata={"method": "keyword"},
            )

        # LLM classification for ambiguous cases
        prompt = get_prompt("safety_filter", self._prompt_version)
        user_content = prompt.template.format_map({"text": text[:1000]})
        messages = [{"role": "user", "content": user_content}]

        llm_text, tokens = await self._llm.chat(
            messages=messages,
            max_tokens=8,
            temperature=0.0,
            stop=["\n", " "],
        )

        raw = llm_text.strip().lower().split()[0] if llm_text.strip() else "warn"
        try:
            llm_label = SafetyLabel(raw)
        except ValueError:
            llm_label = SafetyLabel.WARN

        # Take the worse of keyword + LLM labels
        final = self._merge(keyword_label, llm_label)

        if final != SafetyLabel.SAFE:
            logger.warning(
                "safety_flag",
                label=final,
                keyword=keyword_label,
                llm=llm_label,
                text_preview=text[:80],
            )

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=final.value,
            confidence=0.95 if final == SafetyLabel.SAFE else 0.80,
            tokens_used=tokens,
            metadata={"keyword_label": keyword_label, "llm_label": llm_label},
        )

    def _keyword_check(self, text: str) -> SafetyLabel:
        for p in _BLOCK_PATTERNS:
            if p.search(text):
                return SafetyLabel.BLOCK
        for p in _WARN_PATTERNS:
            if p.search(text):
                return SafetyLabel.WARN
        return SafetyLabel.SAFE

    def _merge(self, a: SafetyLabel, b: SafetyLabel) -> SafetyLabel:
        order = [SafetyLabel.SAFE, SafetyLabel.WARN, SafetyLabel.BLOCK]
        return order[max(order.index(a), order.index(b))]
