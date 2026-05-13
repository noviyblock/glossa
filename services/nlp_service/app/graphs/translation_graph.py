"""LangGraph-based translation workflow with RAG augmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from glossa_common.logging import get_logger
from glossa_common.schemas.nlp import NLPResult, RAGDocument

logger = get_logger(__name__)


class TranslationState(TypedDict):
    session_id: str
    gloss_sequence: str | None
    text: str | None
    domain: str
    context_hint: str | None
    rag_docs: list[RAGDocument]
    rag_context: str
    output_text: str
    tokens_generated: int
    error: str | None


async def retrieve_rag_node(state: TranslationState, retriever: object) -> TranslationState:
    from ..infrastructure.rag_retriever import QdrantRAGRetriever

    r: QdrantRAGRetriever = retriever  # type: ignore[assignment]
    query = state.get("gloss_sequence") or state.get("text") or ""
    if not query:
        return {**state, "rag_docs": [], "rag_context": ""}

    docs = await r.retrieve(query)
    context = r.format_context(docs)
    logger.debug("rag_retrieved", num_docs=len(docs), session=state["session_id"])
    return {**state, "rag_docs": docs, "rag_context": context}


async def generate_node(state: TranslationState, runner: object) -> TranslationState:
    from ..infrastructure.qwen_runner import QwenRunner

    r: QwenRunner = runner  # type: ignore[assignment]
    try:
        text, tokens = await r.generate(
            gloss_sequence=state.get("gloss_sequence"),
            text=state.get("text"),
            domain=state.get("domain", "general"),
            context_hint=state.get("context_hint"),
            rag_context=state.get("rag_context", ""),
        )
        return {**state, "output_text": text, "tokens_generated": tokens, "error": None}
    except Exception as exc:
        logger.exception("generation_failed", session=state["session_id"])
        return {**state, "output_text": "", "tokens_generated": 0, "error": str(exc)}


def build_translation_graph(runner: object, retriever: object):
    """Build a LangGraph StateGraph for the translation pipeline."""
    try:
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(TranslationState)

        workflow.add_node("retrieve_rag", lambda s: retrieve_rag_node(s, retriever))
        workflow.add_node("generate", lambda s: generate_node(s, runner))

        workflow.set_entry_point("retrieve_rag")
        workflow.add_edge("retrieve_rag", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()
    except ImportError:
        logger.warning("langgraph_not_available_using_direct_pipeline")
        return None


class DirectTranslationPipeline:
    """Fallback pipeline when LangGraph is not installed."""

    def __init__(self, runner: object, retriever: object) -> None:
        self._runner = runner
        self._retriever = retriever

    async def ainvoke(self, state: TranslationState) -> TranslationState:
        state = await retrieve_rag_node(state, self._retriever)
        state = await generate_node(state, self._runner)
        return state
