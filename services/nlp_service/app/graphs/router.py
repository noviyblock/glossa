"""Conditional routing logic for the LangGraph orchestrator.

All routing decisions are pure functions of GraphState — no side effects.
"""

from __future__ import annotations

from ..domain.entities import Domain, Intent, SafetyLabel
from .state import GraphState


def route_after_safety_input(state: GraphState) -> str:
    """After input safety check: block or continue to intent classification."""
    if not state.get("input_safety_ok", True):
        return "blocked"
    return "intent"


def route_after_intent(state: GraphState) -> str:
    """Route to the appropriate processing node based on classified intent."""
    intent = state.get("intent", Intent.UNKNOWN.value)
    domain = state.get("domain", Domain.GENERAL.value)

    if intent == Intent.GLOSS_TO_TEXT.value:
        return "rag"

    if intent == Intent.SIMPLIFY.value:
        if domain == Domain.MEDICAL.value:
            return "rag_medical"
        if domain == Domain.BANKING.value:
            return "rag_banking"
        return "rag"

    # QUESTION, CLARIFY, CHITCHAT, TEXT_TO_GLOSS, UNKNOWN → general generation
    return "rag"


def route_after_rag(state: GraphState) -> str:
    """After RAG retrieval, route to the right generation node."""
    intent = state.get("intent", Intent.UNKNOWN.value)
    domain = state.get("domain", Domain.GENERAL.value)

    if intent == Intent.GLOSS_TO_TEXT.value:
        return "gloss_translate"

    if intent == Intent.SIMPLIFY.value:
        if domain == Domain.MEDICAL.value:
            return "medical_simplify"
        if domain == Domain.BANKING.value:
            return "banking_simplify"

    return "response_generate"


def route_after_generation(state: GraphState) -> str:
    """After generation: always run hallucination check then output safety."""
    rag_context = state.get("rag_context", "")
    if rag_context:
        return "hallucination_check"
    return "safety_output"


def route_after_hallucination(state: GraphState) -> str:
    """After hallucination check: always run output safety."""
    return "safety_output"
