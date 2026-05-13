"""Benchmark endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...benchmark.benchmark_service import ASRBenchmarkService
from ...schemas.request import BenchmarkRequest
from ...schemas.response import BenchmarkResponse

router = APIRouter(prefix="/api/v1", tags=["benchmark"])


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    body: BenchmarkRequest,
    request=None,
) -> BenchmarkResponse:
    """Run ASR latency benchmark. Returns RTF and latency percentiles."""
    from fastapi import Request
    # Accessed via request.app.state in the actual handler
    raise HTTPException(status_code=501, detail="Use dependency injection via main.py")


@router.post("/benchmark/run", response_model=BenchmarkResponse)
async def benchmark_run(body: BenchmarkRequest, request_obj=None):
    """Alias handled in main router with app.state injection."""
    raise HTTPException(status_code=501, detail="Handler registered in main.py")
