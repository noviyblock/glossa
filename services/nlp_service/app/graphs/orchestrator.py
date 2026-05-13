"""Main LangGraph orchestrator — builds and compiles the multi-agent graph.

Graph topology:
  safety_input
    ├─ [blocked] → blocked → END
    └─ [continue] → intent → dialogue → token_budget → rag
                                                         ├─ gloss_translate
                                                         ├─ medical_simplify
                                                         ├─ banking_simplify
                                                         └─ response_generate
                                                              ↓
                                                         hallucination_check?
                                                              ↓
                                                         safety_output → END
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import (
    AgentResult,
    Domain,
    Intent,
    NLPRequest,
    NLPResponse,
    SafetyLabel,
    TokenBudget,
)
from .nodes import (
    node_banking_simplify,
    node_blocked,
    node_dialogue,
    node_gloss_translate,
    node_hallucination_check,
    node_medical_simplify,
    node_rag,
    node_response_generate,
    node_safety_input,
    node_safety_output,
    node_token_budget,
    node_intent,
)
from .router import (
    route_after_generation,
    route_after_hallucination,
    route_after_intent,
    route_after_rag,
    route_after_safety_input,
)
from .state import GraphState

logger = get_logger(__name__)


def _bind(fn, agents: dict):
    """Partial-apply agents dict to a node function."""
    async def _wrapped(state: GraphState) -> GraphState:
        return await fn(state, agents)
    _wrapped.__name__ = fn.__name__
    return _wrapped


def build_graph(agents: dict):
    """Compile the LangGraph StateGraph with all agents wired in."""
    try:
        from langgraph.graph import END, StateGraph

        g = StateGraph(GraphState)

        # Register nodes
        g.add_node("safety_input",        _bind(node_safety_input, agents))
        g.add_node("intent",              _bind(node_intent, agents))
        g.add_node("dialogue",            _bind(node_dialogue, agents))
        g.add_node("token_budget",        _bind(node_token_budget, agents))
        g.add_node("rag",                 _bind(node_rag, agents))
        g.add_node("gloss_translate",     _bind(node_gloss_translate, agents))
        g.add_node("medical_simplify",    _bind(node_medical_simplify, agents))
        g.add_node("banking_simplify",    _bind(node_banking_simplify, agents))
        g.add_node("response_generate",   _bind(node_response_generate, agents))
        g.add_node("hallucination_check", _bind(node_hallucination_check, agents))
        g.add_node("safety_output",       _bind(node_safety_output, agents))
        g.add_node("blocked",             _bind(node_blocked, agents))

        # Entry
        g.set_entry_point("safety_input")

        # Conditional routing
        g.add_conditional_edges(
            "safety_input",
            route_after_safety_input,
            {"blocked": "blocked", "intent": "intent"},
        )
        g.add_edge("intent", "dialogue")
        g.add_edge("dialogue", "token_budget")

        # After token budget: always go to RAG
        g.add_edge("token_budget", "rag")

        # After RAG: route to generation node
        g.add_conditional_edges(
            "rag",
            route_after_rag,
            {
                "gloss_translate":   "gloss_translate",
                "medical_simplify":  "medical_simplify",
                "banking_simplify":  "banking_simplify",
                "response_generate": "response_generate",
            },
        )

        # All generation nodes → post-processing
        for gen_node in ("gloss_translate", "medical_simplify", "banking_simplify", "response_generate"):
            g.add_conditional_edges(
                gen_node,
                route_after_generation,
                {
                    "hallucination_check": "hallucination_check",
                    "safety_output":       "safety_output",
                },
            )

        g.add_conditional_edges(
            "hallucination_check",
            route_after_hallucination,
            {"safety_output": "safety_output"},
        )

        g.add_edge("safety_output", END)
        g.add_edge("blocked",       END)

        return g.compile()

    except ImportError:
        logger.warning("langgraph_not_available_using_fallback_executor")
        return None


class NLPOrchestrator:
    """High-level interface that wraps the compiled graph (or fallback).

    Handles:
    - State initialisation from NLPRequest
    - Graph invocation (sync + streaming)
    - Response assembly from final state
    - Post-graph memory update
    """

    def __init__(self, agents: dict, dialogue_manager=None) -> None:
        self._agents = agents
        self._dialogue_manager = dialogue_manager
        self._graph = build_graph(agents)

    async def run(self, request: NLPRequest) -> NLPResponse:
        """Execute the full graph synchronously and return a complete NLPResponse."""
        t0 = time.perf_counter()
        state = self._init_state(request)

        if self._graph is not None:
            final_state: GraphState = await self._graph.ainvoke(state)
        else:
            final_state = await self._fallback_run(state)

        # Persist assistant turn
        if self._dialogue_manager and final_state.get("output_text"):
            await self._dialogue_manager.save_assistant_turn(
                session_id=request.session_id,
                output_text=final_state.get("output_text", ""),
                intent=final_state.get("intent", "unknown"),
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return self._build_response(request, final_state, latency_ms)

    async def stream(self, request: NLPRequest) -> AsyncIterator[str]:
        """Stream tokens from the appropriate generation agent."""
        # Run all pre-generation nodes first (non-streaming)
        t0 = time.perf_counter()
        state = self._init_state(request)

        # Run up to generation node
        pre_state = await self._run_pre_generation(state)
        if not pre_state.get("input_safety_ok", True):
            yield "[Запрос заблокирован]"
            return

        # Stream from the right generation agent
        gen_agent = self._select_generation_agent(pre_state)
        full_output = []
        async for token in gen_agent.stream(pre_state):
            full_output.append(token)
            yield token

        # Post-stream: save memory
        if self._dialogue_manager and full_output:
            await self._dialogue_manager.save_assistant_turn(
                session_id=request.session_id,
                output_text="".join(full_output),
                intent=pre_state.get("intent", "unknown"),
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_state(self, req: NLPRequest) -> GraphState:
        # Infer domain from request or leave as GENERAL
        domain = req.domain.value if req.domain else Domain.GENERAL.value
        return GraphState(
            request_id=req.request_id,
            session_id=req.session_id,
            input_text=req.input_text,
            gloss_sequence=req.gloss_sequence,
            language=req.language,
            stream=req.stream,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            domain=domain,
            intent=Intent.UNKNOWN.value,
            intent_confidence=0.0,
            history_text="",
            turn_count=0,
            generation_budget=req.max_tokens,
            rag_context="",
            rag_docs_used=0,
            safety_label=SafetyLabel.SAFE.value,
            safety_check_text="",
            input_safety_ok=True,
            output_text="",
            tokens_used=0,
            generation_confidence=0.0,
            hallucination_flags=[],
            final_confidence=0.0,
            agent_trace=[],
            error=None,
            latency_ms=0.0,
        )

    def _build_response(
        self, req: NLPRequest, state: GraphState, latency_ms: float
    ) -> NLPResponse:
        from ..domain.entities import AgentResult, AgentStatus
        trace = [
            AgentResult(
                agent=t["agent"],
                status=AgentStatus(t.get("status", "success")),
                latency_ms=t.get("latency_ms", 0),
                confidence=t.get("confidence", 0),
                tokens_used=t.get("tokens_used", 0),
                error=t.get("error"),
            )
            for t in state.get("agent_trace", [])
        ]
        try:
            intent = Intent(state.get("intent", Intent.UNKNOWN.value))
        except ValueError:
            intent = Intent.UNKNOWN
        try:
            domain = Domain(state.get("domain", Domain.GENERAL.value))
        except ValueError:
            domain = Domain.GENERAL
        try:
            safety = SafetyLabel(state.get("safety_label", SafetyLabel.SAFE.value))
        except ValueError:
            safety = SafetyLabel.SAFE

        return NLPResponse(
            request_id=req.request_id,
            session_id=req.session_id,
            output_text=state.get("output_text", ""),
            intent=intent,
            domain=domain,
            confidence=state.get("final_confidence", state.get("generation_confidence", 0.0)),
            safety=safety,
            agent_trace=trace,
            rag_docs_used=state.get("rag_docs_used", 0),
            tokens_used=state.get("tokens_used", 0),
            latency_ms=latency_ms,
            is_fallback=any(
                t.get("status") in ("fallback", "failed") for t in state.get("agent_trace", [])
            ),
            hallucination_flags=state.get("hallucination_flags", []),
        )

    def _select_generation_agent(self, state: GraphState):
        intent = state.get("intent", "")
        domain = state.get("domain", "general")
        if intent == Intent.GLOSS_TO_TEXT.value:
            return self._agents["gloss_translator"]
        if intent == Intent.SIMPLIFY.value and domain == Domain.MEDICAL.value:
            return self._agents["medical_simplifier"]
        if intent == Intent.SIMPLIFY.value and domain == Domain.BANKING.value:
            return self._agents["banking_simplifier"]
        return self._agents["response_generator"]

    async def _run_pre_generation(self, state: GraphState) -> GraphState:
        """Run safety + intent + dialogue + token_budget + RAG nodes, return state."""
        state = await node_safety_input(state, self._agents)
        if not state.get("input_safety_ok", True):
            return state
        state = await node_intent(state, self._agents)
        state = await node_dialogue(state, self._agents)
        state = await node_token_budget(state, self._agents)
        state = await node_rag(state, self._agents)
        return state

    async def _fallback_run(self, state: GraphState) -> GraphState:
        """Sequential fallback when LangGraph is not installed."""
        state = await node_safety_input(state, self._agents)
        if not state.get("input_safety_ok", True):
            return await node_blocked(state, self._agents)
        state = await node_intent(state, self._agents)
        state = await node_dialogue(state, self._agents)
        state = await node_token_budget(state, self._agents)
        state = await node_rag(state, self._agents)

        intent = state.get("intent", "")
        domain = state.get("domain", "general")
        if intent == Intent.GLOSS_TO_TEXT.value:
            state = await node_gloss_translate(state, self._agents)
        elif intent == Intent.SIMPLIFY.value and domain == Domain.MEDICAL.value:
            state = await node_medical_simplify(state, self._agents)
        elif intent == Intent.SIMPLIFY.value and domain == Domain.BANKING.value:
            state = await node_banking_simplify(state, self._agents)
        else:
            state = await node_response_generate(state, self._agents)

        if state.get("rag_context"):
            state = await node_hallucination_check(state, self._agents)
        state = await node_safety_output(state, self._agents)
        return state
