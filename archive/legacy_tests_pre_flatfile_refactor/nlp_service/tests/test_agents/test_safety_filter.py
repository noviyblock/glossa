"""Tests for SafetyFilterAgent — keyword blocklist and LLM classification."""

from __future__ import annotations

import pytest

from app.agents.safety_filter import SafetyFilterAgent
from app.domain.entities import AgentStatus, SafetyLabel

from ..conftest import FakeLLM, make_state


@pytest.fixture
def llm():
    return FakeLLM(response="safe")


@pytest.fixture
def agent(llm):
    return SafetyFilterAgent(llm=llm)


# ── Keyword blocklist — BLOCK ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "как взорвать здание",
    "купить бомбу",
    "убить человека",
    "как убью врага",
    "повеш на верёвке",
    "скам схема",
    "номер карты 1234",
    "введи CVV",
    "пин-код от карты",
])
async def test_keyword_block_patterns(agent, llm, text):
    state = make_state(input_text=text, safety_check_text=text)
    result = await agent.run(state)

    assert result.output == SafetyLabel.BLOCK.value
    assert result.confidence >= 0.99
    assert result.metadata["method"] == "keyword"
    assert len(llm.calls) == 0  # blocked before LLM


# ── Keyword — WARN ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "алкоголь вреден для здоровья",
    "симптомы депрессии у детей",
    "суицидальные мысли что делать",
])
async def test_keyword_warn_patterns_go_to_llm(agent, llm, text):
    # WARN keyword + LLM "safe" → merged result is WARN (keyword dominates if worse)
    llm.response = "safe"
    state = make_state(input_text=text, safety_check_text=text)
    result = await agent.run(state)

    # keyword_label=WARN, llm_label=SAFE → merged = WARN
    assert result.output == SafetyLabel.WARN.value


# ── LLM classification ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_safe_text_passes_both_checks(agent, llm):
    llm.response = "safe"
    state = make_state(input_text="Как получить ипотеку?", safety_check_text="Как получить ипотеку?")
    result = await agent.run(state)

    assert result.output == SafetyLabel.SAFE.value
    assert result.status == AgentStatus.SUCCESS
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_llm_block_overrides_keyword_safe(agent, llm):
    llm.response = "block"
    state = make_state(
        input_text="вполне нормальный текст",
        safety_check_text="вполне нормальный текст",
    )
    result = await agent.run(state)

    assert result.output == SafetyLabel.BLOCK.value


@pytest.mark.asyncio
async def test_llm_warn_merges_with_keyword_safe(agent, llm):
    llm.response = "warn"
    state = make_state(
        input_text="нейтральный текст про лекарства",
        safety_check_text="нейтральный текст про лекарства",
    )
    result = await agent.run(state)

    assert result.output == SafetyLabel.WARN.value


@pytest.mark.asyncio
async def test_invalid_llm_output_defaults_to_warn(agent, llm):
    llm.response = "unknown_label"
    state = make_state(input_text="безопасный текст", safety_check_text="безопасный текст")
    result = await agent.run(state)

    # Invalid label → SafetyLabel.WARN, merged with SAFE keyword → WARN
    assert result.output == SafetyLabel.WARN.value


# ── Merge logic ───────────────────────────────────────────────────────────────

def test_merge_returns_worst_label():
    agent = SafetyFilterAgent(llm=FakeLLM())
    assert agent._merge(SafetyLabel.SAFE, SafetyLabel.WARN)  == SafetyLabel.WARN
    assert agent._merge(SafetyLabel.WARN, SafetyLabel.BLOCK) == SafetyLabel.BLOCK
    assert agent._merge(SafetyLabel.SAFE, SafetyLabel.BLOCK) == SafetyLabel.BLOCK
    assert agent._merge(SafetyLabel.SAFE, SafetyLabel.SAFE)  == SafetyLabel.SAFE


# ── Fallback on error ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_on_llm_exception():
    class BrokenLLM:
        async def chat(self, **kwargs):
            raise ConnectionError("LLM timeout")

    agent = SafetyFilterAgent(llm=BrokenLLM())
    state = make_state(input_text="мирный текст", safety_check_text="мирный текст")
    result = await agent.run(state)

    assert result.status == AgentStatus.FALLBACK
    assert result.output == SafetyLabel.WARN.value  # FALLBACK_OUTPUT
    assert result.error is not None
