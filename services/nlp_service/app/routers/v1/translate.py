import time

from fastapi import APIRouter, HTTPException, Request, status

from glossa_common.schemas.nlp import NLPRequest, NLPResult

router = APIRouter()


@router.post(
    "/translate",
    response_model=NLPResult,
    status_code=status.HTTP_200_OK,
    summary="Translate gloss sequence → text or text → gloss sequence",
)
async def translate(request: NLPRequest, http_request: Request) -> NLPResult:
    graph = http_request.app.state.graph
    t0 = time.perf_counter()

    state = {
        "session_id": str(request.session_id),
        "gloss_sequence": request.gloss_sequence,
        "text": request.text,
        "domain": request.domain,
        "context_hint": request.context_hint,
        "rag_docs": [],
        "rag_context": "",
        "output_text": "",
        "tokens_generated": 0,
        "error": None,
    }

    try:
        final_state = await graph.ainvoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NLP pipeline error: {exc}") from exc

    if final_state.get("error"):
        raise HTTPException(status_code=500, detail=final_state["error"])

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Determine output mode
    is_rsl_to_text = request.gloss_sequence is not None
    return NLPResult(
        session_id=request.session_id,
        text=final_state["output_text"] if is_rsl_to_text else request.text or "",
        gloss_sequence=None if is_rsl_to_text else final_state["output_text"],
        tokens_generated=final_state["tokens_generated"],
        rag_documents=final_state.get("rag_docs", []),
        model_name="Qwen2-1.5B-Instruct",
        processing_time_ms=round(elapsed_ms, 2),
    )
