"""Benchmark endpoints — trigger runs, compare models, query MLflow."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...dependencies import get_benchmark_service
from ...schemas.model import benchmark_to_response
from ...schemas.request import BenchmarkRequest
from ...schemas.response import BenchmarkCompareResponse, BenchmarkResultResponse
from ...service.benchmark_service import BenchmarkService

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])


@router.post("/run", response_model=BenchmarkCompareResponse)
async def run_benchmark(
    body: BenchmarkRequest,
    svc: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkCompareResponse:
    """Run latency benchmarks on one or more models and optionally compare them."""
    try:
        results = await svc.compare(
            model_ids=body.model_ids,
            n_samples=body.n_samples,
            batch_size=body.batch_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not results:
        raise HTTPException(status_code=404, detail="No results — check model IDs")

    winner = results[0].model_id if results else None
    return BenchmarkCompareResponse(
        results=[BenchmarkResultResponse(**benchmark_to_response(r)) for r in results],
        winner_model_id=winner,
    )


@router.get("/compare", response_model=BenchmarkCompareResponse)
async def compare_from_mlflow(
    model_ids: str,
    svc: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkCompareResponse:
    """Fetch latest benchmark results from MLflow for comma-separated model IDs."""
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="model_ids query param required")

    rows = svc.get_mlflow_comparison(ids)
    if not rows:
        return BenchmarkCompareResponse(results=[], winner_model_id=None)

    rows.sort(key=lambda r: r.get("latency_p95_ms") or float("inf"))
    winner = rows[0].get("model_id") if rows else None

    results = [
        BenchmarkResultResponse(
            model_id=r.get("model_id", ""),
            model_version="",
            backend=r.get("backend", ""),
            quantization=r.get("quantization", ""),
            n_samples=0,
            batch_size=0,
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=r.get("latency_p95_ms") or 0.0,
            latency_p99_ms=0.0,
            latency_max_ms=0.0,
            throughput_rps=r.get("throughput_rps") or 0.0,
            warmup_ms=0.0,
            mlflow_run_id=r.get("run_id"),
        )
        for r in rows
    ]
    return BenchmarkCompareResponse(results=results, winner_model_id=winner)
