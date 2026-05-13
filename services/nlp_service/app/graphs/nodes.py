"""LangGraph node wrappers — each calls an agent and merges result into state."""

from __future__ import annotations

import json
import time
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import Domain, Intent, SafetyLabel, TokenBudget
from .state import GraphState

logger = get_logger(__name__)

_BLOCKED_RESPONSE = "Запрос заблокирован системой безопасности. Пожалуйста, переформулируйте вопрос."


def _record(state: dict, result) -> None:
    """Append agent result to the trace list."""
    trace = state.get("agent_trace", [])
    trace.append({
        "agent": result.agent,
        "status": result.status,
        "latency_ms": round(result.latency_ms, 1),
        "confidence": round(result.confidence, 3),
        "tokens_used": result.tokens_used,
        "error": result.error,
    })
    state["agent_trace"] = trace


# ── Node functions ────────────────────────────────────────────────────────────

async def node_safety_input(state: GraphState, agents: dict) -> GraphState:
    """Pre-generation safety screen on the raw user input."""
    agent = agents["safety_filter"]
    result = await agent.run({**state, "safety_check_text": state.get("input_text", "")})
    _record(state, result)

    label = SafetyLabel(result.output) if result.output in SafetyLabel._value2member_map_ else SafetyLabel.WARN
    return {
        **state,
        "safety_label": label.value,
        "input_safety_ok": label != SafetyLabel.BLOCK,
    }


async def node_intent(state: GraphState, agents: dict) -> GraphState:
    """Classify user intent."""
    agent = agents["intent_classifier"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "intent": result.output,
        "intent_confidence": result.confidence,
    }


async def node_dialogue(state: GraphState, agents: dict) -> GraphState:
    """Load conversation history and update memory."""
    agent = agents["dialogue_manager"]
    result = await agent.run(state)
    _record(state, result)

    meta = result.metadata
    return {
        **state,
        "history_text": result.output,
        "turn_count": meta.get("turn_count", 0),
    }


async def node_token_budget(state: GraphState, agents: dict) -> GraphState:
    """Compute token budget for this request."""
    max_tokens: int = state.get("max_tokens", 512)
    history_len = len(state.get("history_text", "")) // 4  # rough char→token
    budget = TokenBudget(
        total=max_tokens,
        system_reserved=128,
        rag_reserved=min(384, max_tokens // 4),
        history_reserved=min(history_len + 32, max_tokens // 4),
    )
    generation_budget = budget.generation_budget
    return {
        **state,
        "token_budget": budget,
        "generation_budget": generation_budget,
    }


async def node_rag(state: GraphState, agents: dict) -> GraphState:
    """Retrieve relevant documents from the vector store."""
    agent = agents["rag_retriever"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "rag_context": result.output,
        "rag_docs_used": result.metadata.get("n_docs", 0),
    }


async def node_gloss_translate(state: GraphState, agents: dict) -> GraphState:
    agent = agents["gloss_translator"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "output_text": result.output,
        "tokens_used": state.get("tokens_used", 0) + result.tokens_used,
        "generation_confidence": result.confidence,
    }


async def node_medical_simplify(state: GraphState, agents: dict) -> GraphState:
    agent = agents["medical_simplifier"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "output_text": result.output,
        "tokens_used": state.get("tokens_used", 0) + result.tokens_used,
        "generation_confidence": result.confidence,
    }


async def node_banking_simplify(state: GraphState, agents: dict) -> GraphState:
    agent = agents["banking_simplifier"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "output_text": result.output,
        "tokens_used": state.get("tokens_used", 0) + result.tokens_used,
        "generation_confidence": result.confidence,
    }


async def node_response_generate(state: GraphState, agents: dict) -> GraphState:
    agent = agents["response_generator"]
    result = await agent.run(state)
    _record(state, result)

    return {
        **state,
        "output_text": result.output,
        "tokens_used": state.get("tokens_used", 0) + result.tokens_used,
        "generation_confidence": result.confidence,
    }


async def node_hallucination_check(state: GraphState, agents: dict) -> GraphState:
    agent = agents["hallucination_checker"]
    result = await agent.run(state)
    _record(state, result)

    try:
        flags = json.loads(result.output) if result.output else []
    except json.JSONDecodeError:
        flags = []

    gen_conf = state.get("generation_confidence", 0.8)
    final_conf = gen_conf * result.confidence
    return {
        **state,
        "hallucination_flags": flags,
        "final_confidence": final_conf,
    }


async def node_safety_output(state: GraphState, agents: dict) -> GraphState:
    """Post-generation safety screen on the produced output."""
    agent = agents["safety_filter"]
    result = await agent.run({**state, "safety_check_text": state.get("output_text", "")})
    _record(state, result)

    label = SafetyLabel(result.output) if result.output in SafetyLabel._value2member_map_ else SafetyLabel.WARN
    output = state.get("output_text", "")
    if label == SafetyLabel.BLOCK:
        output = _BLOCKED_RESPONSE
        logger.warning("safety_output_blocked", session_id=state.get("session_id", ""))

    return {
        **state,
        "output_text": output,
        "safety_label": label.value,
    }


async def node_blocked(state: GraphState, agents: dict) -> GraphState:
    """Terminal node for blocked inputs."""
    return {
        **state,
        "output_text": _BLOCKED_RESPONSE,
        "final_confidence": 0.0,
        "hallucination_flags": [],
    }
