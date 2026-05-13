"""LangGraph state definition — the single mutable context flowing through all nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    request_id: str
    session_id: str
    input_text: str
    gloss_sequence: str | None
    language: str
    stream: bool
    max_tokens: int
    temperature: float

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str             # Intent enum value
    domain: str             # Domain enum value
    intent_confidence: float

    # ── Memory ────────────────────────────────────────────────────────────────
    history_text: str
    turn_count: int

    # ── Token budget ──────────────────────────────────────────────────────────
    generation_budget: int
    token_budget: Any       # TokenBudget instance

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_context: str
    rag_docs_used: int

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_label: str       # SafetyLabel enum value
    safety_check_text: str
    input_safety_ok: bool

    # ── Generation output ─────────────────────────────────────────────────────
    output_text: str
    tokens_used: int
    generation_confidence: float

    # ── Post-processing ───────────────────────────────────────────────────────
    hallucination_flags: list[str]
    final_confidence: float

    # ── Telemetry ─────────────────────────────────────────────────────────────
    agent_trace: list[dict]
    error: str | None
    latency_ms: float
