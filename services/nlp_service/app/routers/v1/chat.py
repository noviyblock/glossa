"""Main chat and translation REST + streaming endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...domain.entities import Domain, Intent, NLPRequest
from ...schemas.request import NLPChatRequest, SimplifyRequest, TranslateGlossRequest
from ...schemas.response import (
    AgentTraceItem,
    MessageResponse,
    NLPChatResponse,
    SimplifyResponse,
    TranslateGlossResponse,
)

router = APIRouter(prefix="/api/v1", tags=["nlp"])


def _get_orchestrator(request: Request):
    return request.app.state.orchestrator


def _get_memory_store(request: Request):
    return request.app.state.memory_store


@router.post("/chat", response_model=NLPChatResponse)
async def chat(
    body: NLPChatRequest,
    request: Request,
) -> NLPChatResponse:
    """Full multi-agent NLP pipeline: intent → RAG → generation → safety."""
    orchestrator = _get_orchestrator(request)
    nlp_req = NLPRequest(
        session_id=body.session_id,
        input_text=body.input_text,
        gloss_sequence=body.gloss_sequence,
        language=body.language,
        domain=body.domain,
        stream=False,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )
    try:
        resp = await orchestrator.run(nlp_req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return NLPChatResponse(
        request_id=resp.request_id,
        session_id=resp.session_id,
        output_text=resp.output_text,
        intent=resp.intent.value,
        domain=resp.domain.value,
        confidence=round(resp.confidence, 4),
        safety=resp.safety.value,
        rag_docs_used=resp.rag_docs_used,
        tokens_used=resp.tokens_used,
        latency_ms=round(resp.latency_ms, 2),
        is_fallback=resp.is_fallback,
        hallucination_flags=resp.hallucination_flags,
        agent_trace=[
            AgentTraceItem(
                agent=t.agent,
                status=t.status.value,
                latency_ms=round(t.latency_ms, 1),
                confidence=round(t.confidence, 3),
                tokens_used=t.tokens_used,
                error=t.error,
            )
            for t in resp.agent_trace
        ],
    )


@router.post("/chat/stream")
async def chat_stream(body: NLPChatRequest, request: Request):
    """Streaming response — yields tokens as Server-Sent Events."""
    orchestrator = _get_orchestrator(request)
    nlp_req = NLPRequest(
        session_id=body.session_id,
        input_text=body.input_text,
        gloss_sequence=body.gloss_sequence,
        language=body.language,
        domain=body.domain,
        stream=True,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    async def _sse():
        async for token in orchestrator.stream(nlp_req):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@router.post("/translate/gloss", response_model=TranslateGlossResponse)
async def translate_gloss(body: TranslateGlossRequest, request: Request) -> TranslateGlossResponse:
    """Dedicated gloss-to-text translation endpoint (no intent classification overhead)."""
    orchestrator = _get_orchestrator(request)
    t0 = time.perf_counter()
    nlp_req = NLPRequest(
        session_id=body.session_id,
        input_text=body.gloss_sequence,
        gloss_sequence=body.gloss_sequence,
        domain=body.domain,
        stream=False,
        max_tokens=body.max_tokens,
    )
    try:
        resp = await orchestrator.run(nlp_req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TranslateGlossResponse(
        session_id=resp.session_id,
        gloss_sequence=body.gloss_sequence,
        translated_text=resp.output_text,
        domain=resp.domain.value,
        confidence=round(resp.confidence, 4),
        rag_docs_used=resp.rag_docs_used,
        inference_ms=round(resp.latency_ms, 2),
        is_fallback=resp.is_fallback,
    )


@router.post("/simplify", response_model=SimplifyResponse)
async def simplify(body: SimplifyRequest, request: Request) -> SimplifyResponse:
    """Simplify medical or banking text for RSL users."""
    orchestrator = _get_orchestrator(request)
    t0 = time.perf_counter()
    nlp_req = NLPRequest(
        session_id=body.session_id,
        input_text=body.text,
        domain=Domain(body.domain),
        stream=False,
        max_tokens=body.max_tokens,
    )
    # Force simplify intent
    try:
        resp = await orchestrator.run(nlp_req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SimplifyResponse(
        session_id=resp.session_id,
        original_text=body.text,
        simplified_text=resp.output_text,
        domain=resp.domain.value,
        confidence=round(resp.confidence, 4),
        inference_ms=round(resp.latency_ms, 2),
    )


@router.post("/conversation/reset", response_model=MessageResponse)
async def reset_conversation(session_id: str, request: Request) -> MessageResponse:
    store = _get_memory_store(request)
    await store.delete(session_id)
    return MessageResponse(message=f"Conversation reset for session '{session_id}'")
