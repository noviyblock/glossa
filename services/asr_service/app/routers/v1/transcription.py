import time
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from glossa_common.schemas.asr import ASRRequest, ASRResult

import orjson

router = APIRouter()


@router.post(
    "/transcribe",
    response_model=ASRResult,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio bytes to text (synchronous)",
)
async def transcribe(request: ASRRequest, http_request: Request) -> ASRResult:
    runner = http_request.app.state.runner
    t0 = time.perf_counter()
    try:
        result = await runner.transcribe(request.audio_bytes, request.sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return ASRResult(
        session_id=request.session_id,
        text=result["text"],
        language=result["language"],
        language_probability=result["language_probability"],
        duration_seconds=result["duration"],
        words=result.get("words", []),
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/transcribe/stream",
    summary="Stream ASR chunks via Server-Sent Events",
)
async def transcribe_stream(http_request: Request) -> StreamingResponse:
    runner = http_request.app.state.runner
    audio_bytes = await http_request.body()
    session_id_header = http_request.headers.get("X-Session-ID", "unknown")

    async def _generate():
        chunk_index = 0
        async for chunk in runner.transcribe_stream(audio_bytes):
            chunk["session_id"] = session_id_header
            yield orjson.dumps(chunk) + b"\n"
            chunk_index += 1
        yield orjson.dumps({"is_final": True, "chunk_index": chunk_index, "session_id": session_id_header}) + b"\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")
