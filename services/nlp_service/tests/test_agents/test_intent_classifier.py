"""Tests for IntentClassifierAgent."""

from __future__ import annotations

import pytest

from app.agents.intent_classifier import IntentClassifierAgent
from app.domain.entities import AgentStatus, Intent

from ..conftest import FakeLLM, make_state


@pytest.fixture
def llm():
    return FakeLLM(response="gloss_to_text")


@pytest.fixture
def agent(llm):
    return IntentClassifierAgent(llm=llm)


# ── Heuristic paths ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_gloss_sequence_bypasses_llm(agent, llm):
    state = make_state(gloss_sequence="ПРИВЕТ МИР", input_text="hello")
    result = await agent.run(state)

    assert result.status == AgentStatus.SUCCESS
    assert result.output == Intent.GLOSS_TO_TEXT.value
    assert result.confidence >= 0.99
    assert result.metadata["method"] == "heuristic_gloss"
    assert len(llm.calls) == 0  # LLM not called


@pytest.mark.asyncio
async def test_all_caps_text_detected_as_gloss(agent, llm):
    state = make_state(input_text="ДОБРЫЙ ДЕНЬ")
    result = await agent.run(state)

    assert result.output == Intent.GLOSS_TO_TEXT.value
    assert result.metadata["method"] == "heuristic_caps"
    assert result.confidence >= 0.85
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_normal_text_uses_llm(agent, llm):
    llm.response = "question"
    state = make_state(input_text="Что такое страхование?")
    result = await agent.run(state)

    assert result.output == Intent.QUESTION.value
    assert result.metadata["method"] == "llm"
    assert len(llm.calls) == 1


# ── LLM output parsing ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_llm_output_maps_to_unknown(agent, llm):
    llm.response = "nonsense_value"
    state = make_state(input_text="Что-то непонятное")
    result = await agent.run(state)

    assert result.output == Intent.UNKNOWN.value
    assert result.confidence == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_llm_simplify_intent_recognized(agent, llm):
    llm.response = "simplify"
    state = make_state(input_text="Упрости медицинский текст")
    result = await agent.run(state)

    assert result.output == Intent.SIMPLIFY.value
    assert result.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_llm_response_with_extra_whitespace(agent, llm):
    llm.response = "  gloss_to_text  \n"
    state = make_state(input_text="нормальный текст")
    result = await agent.run(state)

    assert result.output == Intent.GLOSS_TO_TEXT.value


# ── Confidence ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_intent_has_high_confidence(agent, llm):
    for intent_val in [i.value for i in Intent if i != Intent.UNKNOWN]:
        llm.response = intent_val
        state = make_state(input_text="какой-то текст")
        result = await agent.run(state)
        assert result.confidence >= 0.90, f"Expected high confidence for {intent_val}"


# ── Fallback on LLM error ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_on_llm_exception():
    class BrokenLLM:
        async def chat(self, **kwargs):
            raise RuntimeError("LLM unavailable")

    agent = IntentClassifierAgent(llm=BrokenLLM())
    state = make_state(input_text="какой-то текст")
    result = await agent.run(state)

    assert result.status == AgentStatus.FALLBACK
    assert result.confidence == 0.0
    assert result.error is not None
