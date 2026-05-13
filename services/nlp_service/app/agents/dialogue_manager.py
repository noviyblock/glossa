"""Agent 5 — Dialogue context manager.

Maintains conversation memory: loads history, updates summary,
builds history_text for downstream agents.
"""

from __future__ import annotations

import json
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import (
    AgentResult,
    AgentStatus,
    ConversationMemory,
    ConversationTurn,
    Domain,
    Intent,
)
from ..domain.prompts import get_prompt
from .base import BaseAgent

logger = get_logger(__name__)

_SUMMARIZE_AFTER_TURNS = 10   # compress history after this many turns
_MAX_HISTORY_CHARS     = 1500


class DialogueManagerAgent(BaseAgent):
    """Loads, enriches, and saves conversation context."""

    @property
    def name(self) -> str:
        return "dialogue_manager"

    def __init__(self, llm, memory_store, prompt_version: str = "1.0") -> None:
        super().__init__(llm, prompt_version)
        self._store = memory_store

    async def _execute(self, state: dict[str, Any]) -> AgentResult:
        session_id: str = state.get("session_id", "")
        input_text: str = state.get("input_text", "")
        intent_str: str = state.get("intent", Intent.UNKNOWN.value)
        domain_str: str = state.get("domain", Domain.GENERAL.value)

        # Load or create memory
        memory: ConversationMemory | None = await self._store.load(session_id)
        if memory is None:
            memory = ConversationMemory(session_id=session_id)

        # Trigger summarization if history is long
        if len(memory.turns) >= _SUMMARIZE_AFTER_TURNS:
            await self._compress(memory)

        # Build history text for downstream agents
        history_text = self._build_history_text(memory)

        # Add current user turn to memory
        try:
            intent = Intent(intent_str)
            domain = Domain(domain_str)
        except ValueError:
            intent = Intent.UNKNOWN
            domain = Domain.GENERAL

        memory.add_turn(
            role="user",
            content=input_text,
            intent=intent,
            domain=domain,
        )
        memory.dominant_domain = domain
        await self._store.save(memory)

        return AgentResult(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            output=history_text,
            confidence=1.0,
            metadata={
                "turn_count": len(memory.turns),
                "has_summary": bool(memory.summary),
                "dominant_domain": domain,
            },
        )

    async def save_assistant_turn(
        self, session_id: str, output_text: str, intent: str
    ) -> None:
        """Call after response is generated to persist the assistant turn."""
        memory = await self._store.load(session_id)
        if memory is None:
            return
        memory.add_turn(role="assistant", content=output_text)
        await self._store.save(memory)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_history_text(self, memory: ConversationMemory) -> str:
        parts: list[str] = []
        if memory.summary:
            parts.append(f"[Резюме]: {memory.summary}")
        for turn in memory.recent_turns(6):
            label = "Пользователь" if turn.role == "user" else "Ассистент"
            parts.append(f"{label}: {turn.content}")
        text = "\n".join(parts)
        # Hard cap to avoid exceeding token budget
        if len(text) > _MAX_HISTORY_CHARS:
            text = text[-_MAX_HISTORY_CHARS:]
        return text or "Начало диалога"

    async def _compress(self, memory: ConversationMemory) -> None:
        """Summarize old turns and replace them with the summary."""
        old_turns = memory.turns[:-4]  # keep last 4 turns in full
        if not old_turns:
            return

        history_str = "\n".join(
            f"{'Пользователь' if t.role == 'user' else 'Ассистент'}: {t.content}"
            for t in old_turns
        )
        prompt = get_prompt("dialogue_summarizer", self._prompt_version)
        user_content = prompt.template.format_map({"history": history_str})
        try:
            summary, _ = await self._llm.chat(
                messages=[{"role": "user", "content": user_content}],
                max_tokens=128,
                temperature=0.0,
                stop=None,
            )
            memory.summary = summary.strip()
            memory.turns = memory.turns[-4:]
            logger.info(
                "dialogue_compressed",
                session_id=memory.session_id,
                summary_len=len(memory.summary),
            )
        except Exception:
            logger.exception("dialogue_compress_failed", session_id=memory.session_id)
