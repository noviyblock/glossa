"""Integration tests for NLPOrchestrator fallback path (no LangGraph required)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.entities import (
    AgentResult,
    AgentStatus,
    Domain,
    Intent,
    NLPRequest,
    SafetyLabel,
)
from app.graphs.orchestrator import NLPOrchestrator

from ..conftest import FakeLLM, FakeMemoryStore, FakeRetriever, make_state


# ── Stub agents ───────────────────────────────────────────────────────────────

def _stub_agent(name: str, output: str, confidence: float = 0.9, tokens: int = 10):
    """Return a minimal agent mock that satisfies the AgentResult contract."""
    agent = MagicMock()
    agent.name = name
    result = AgentResult(
        agent=name,
        status=AgentStatus.SUCCESS,
        output=output,
        confidence=confidence,
        tokens_used=tokens,
    )
    agent.run = AsyncMock(return_value=result)

    async def _stream(state):
        for token in output.split():
            yield token

    agent.stream = _stream
    return agent


def _make_agents(
    intent: str = "question",
    safety: str = "safe",
    output_text: str = "Это ответ на ваш вопрос.",
) -> dict:
    return {
        "intent_classifier":     _stub_agent("intent_classifier", intent, 0.95),
        "safety_filter":         _stub_agent("safety_filter", safety, 0.99),
        "dialogue_manager":      _stub_agent("dialogue_manager", "", 1.0),
        "rag_retriever":         _stub_agent("rag_retriever", "", 1.0),
        "gloss_translator":      _stub_agent("gloss_translator", output_text, 0.88),
        "medical_simplifier":    _stub_agent("medical_simplifier", output_text, 0.90),
        "banking_simplifier":    _stub_agent("banking_simplifier", output_text, 0.90),
        "response_generator":    _stub_agent("response_generator", output_text, 0.85),
        "hallucination_checker": _stub_agent("hallucination_checker", "[]", 1.0),
    }


@pytest.fixture
def memory_store():
    return FakeMemoryStore()


# ── Orchestrator with LangGraph disabled ──────────────────────────────────────

def _make_orchestrator(agents, dialogue_manager=None):
    with patch("app.graphs.orchestrator.build_graph", return_value=None):
        orch = NLPOrchestrator(agents=agents, dialogue_manager=dialogue_manager)
    return orch


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_basic_question_returns_nlp_response():
    agents = _make_agents(intent="question", output_text="Ответ готов.")
    orch = _make_orchestrator(agents)

    req = NLPRequest(input_text="Что такое RSL?", session_id="s1")
    resp = await orch.run(req)

    assert resp.output_text == "Ответ готов."
    assert resp.session_id == "s1"
    assert resp.request_id == req.request_id


@pytest.mark.asyncio
async def test_blocked_input_returns_blocked_output():
    agents = _make_agents(safety="block", intent="question")
    orch = _make_orchestrator(agents)

    req = NLPRequest(input_text="Запрещённый текст", session_id="s2")
    resp = await orch.run(req)

    assert resp.safety == SafetyLabel.BLOCK
    assert resp.output_text  # blocked node emits a "request blocked" message
    # response generator should NOT have been called for a blocked request
    agents["response_generator"].run.assert_not_awaited()


@pytest.mark.asyncio
async def test_gloss_intent_routes_to_gloss_translator():
    output = "Привет, мир!"
    agents = _make_agents(intent="gloss_to_text", output_text=output)
    orch = _make_orchestrator(agents)

    req = NLPRequest(
        gloss_sequence="ПРИВЕТ МИР",
        input_text="ПРИВЕТ МИР",
        session_id="s3",
    )
    resp = await orch.run(req)

    assert resp.output_text == output
    agents["gloss_translator"].run.assert_awaited()


@pytest.mark.asyncio
async def test_medical_simplify_routes_correctly():
    output = "Упрощённый текст."
    agents = _make_agents(intent="simplify", output_text=output)
    orch = _make_orchestrator(agents)

    req = NLPRequest(
        input_text="Упрости",
        domain=Domain.MEDICAL,
        session_id="s4",
    )
    resp = await orch.run(req)

    assert resp.output_text == output
    agents["medical_simplifier"].run.assert_awaited()


@pytest.mark.asyncio
async def test_banking_simplify_routes_correctly():
    output = "Банковский текст упрощён."
    agents = _make_agents(intent="simplify", output_text=output)
    orch = _make_orchestrator(agents)

    req = NLPRequest(
        input_text="Упрости банковские условия",
        domain=Domain.BANKING,
        session_id="s5",
    )
    resp = await orch.run(req)

    assert resp.output_text == output
    agents["banking_simplifier"].run.assert_awaited()


@pytest.mark.asyncio
async def test_dialogue_manager_persists_turn(memory_store):
    output = "Ответ сохранён."
    agents = _make_agents(intent="question", output_text=output)
    dm = agents["dialogue_manager"]
    orch = _make_orchestrator(agents, dialogue_manager=dm)

    req = NLPRequest(input_text="Сохрани это", session_id="s6")
    await orch.run(req)

    dm.run.assert_awaited()


@pytest.mark.asyncio
async def test_streaming_yields_tokens():
    output = "Это потоковый ответ."
    agents = _make_agents(intent="question", output_text=output)
    orch = _make_orchestrator(agents)

    req = NLPRequest(input_text="Стриминг?", session_id="s7", stream=True)

    tokens = []
    async for token in orch.stream(req):
        tokens.append(token)

    assert len(tokens) > 0
    assert "".join(tokens) != ""


@pytest.mark.asyncio
async def test_is_fallback_true_when_agent_fails():
    agents = _make_agents(intent="question", output_text="Ответ.")
    # Make response generator return FALLBACK
    fail_result = AgentResult(
        agent="response_generator",
        status=AgentStatus.FALLBACK,
        output="",
        confidence=0.0,
        error="LLM unavailable",
    )
    agents["response_generator"].run = AsyncMock(return_value=fail_result)
    orch = _make_orchestrator(agents)

    req = NLPRequest(input_text="Любой вопрос", session_id="s8")
    resp = await orch.run(req)

    assert resp.is_fallback is True


@pytest.mark.asyncio
async def test_response_has_agent_trace():
    agents = _make_agents(intent="question", output_text="Трассировка.")
    orch = _make_orchestrator(agents)

    req = NLPRequest(input_text="Вопрос", session_id="s9")
    resp = await orch.run(req)

    assert isinstance(resp.agent_trace, list)
    # Fallback run executes multiple nodes → trace should be non-empty
    assert len(resp.agent_trace) >= 1
