"""Admin endpoints — prompt versioning, graph inspection, cache stats."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...domain.prompts import PROMPT_REGISTRY, PromptVersion, list_prompts
from ...schemas.request import AddPromptRequest
from ...schemas.response import MessageResponse, PromptListResponse

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/prompts", response_model=PromptListResponse)
async def list_prompt_versions() -> PromptListResponse:
    """List all registered prompt versions."""
    return PromptListResponse(prompts=list_prompts())


@router.post("/prompts", response_model=MessageResponse, status_code=201)
async def register_prompt(body: AddPromptRequest) -> MessageResponse:
    """Register a new prompt template at runtime."""
    key = f"{body.agent}:{body.version}"
    if key in PROMPT_REGISTRY:
        raise HTTPException(status_code=409, detail=f"Prompt '{key}' already exists")
    PROMPT_REGISTRY[key] = PromptVersion(
        name=body.agent,
        version=body.version,
        template=body.template,
        language=body.language,
    )
    return MessageResponse(message=f"Prompt '{key}' registered")


@router.get("/graph")
async def graph_info(request: Request) -> dict:
    """Return graph topology metadata."""
    orchestrator = request.app.state.orchestrator
    has_graph = orchestrator._graph is not None
    return {
        "langgraph_available": has_graph,
        "agents": list(orchestrator._agents.keys()),
        "mode": "langgraph" if has_graph else "fallback_sequential",
    }


@router.get("/memory/{session_id}")
async def get_memory_stats(session_id: str, request: Request) -> dict:
    """Return conversation memory stats for a session."""
    store = request.app.state.memory_store
    memory = await store.load(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "turn_count": len(memory.turns),
        "dominant_domain": memory.dominant_domain,
        "has_summary": bool(memory.summary),
        "summary_preview": memory.summary[:100] if memory.summary else None,
    }


@router.delete("/memory/{session_id}", response_model=MessageResponse)
async def clear_memory(session_id: str, request: Request) -> MessageResponse:
    store = request.app.state.memory_store
    await store.delete(session_id)
    return MessageResponse(message=f"Memory cleared for '{session_id}'")
