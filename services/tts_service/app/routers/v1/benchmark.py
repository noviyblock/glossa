"""Benchmark endpoint — measures TTS synthesis latency and RTF."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...schemas.request import BenchmarkRequest
from ...schemas.response import BenchmarkResponse

router = APIRouter()


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Run TTS latency benchmark — returns P50/P95/P99 and RTF",
)
async def benchmark(body: BenchmarkRequest, request: Request) -> BenchmarkResponse:
    pipeline = request.app.state.pipeline
    try:
        result = await pipeline.benchmark(
            text=body.text,
            voice_id=body.voice_id,
            n=body.n_runs,
            fmt=body.format,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return BenchmarkResponse(
        n_runs=result.n_runs,
        text_len=result.text_len,
        voice_id=result.voice_id,
        sample_rate=result.sample_rate,
        p50_ms=result.p50_ms,
        p95_ms=result.p95_ms,
        p99_ms=result.p99_ms,
        min_ms=result.min_ms,
        max_ms=result.max_ms,
        rtf=result.rtf,
        chars_per_sec=result.chars_per_sec,
        format=result.format,
    )
