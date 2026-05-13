"""Cache management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from ...schemas.request import PreloadRequest
from ...schemas.response import CacheStatsResponse, PreloadResponse

router = APIRouter()


def _get_pipeline(request: Request):
    return request.app.state.pipeline


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics — entries, hit rate, byte usage",
)
async def cache_stats(request: Request) -> CacheStatsResponse:
    pipeline = _get_pipeline(request)
    stats = await pipeline._cache.stats()
    return CacheStatsResponse(
        l1=stats.get("l1", stats),
        l2=stats.get("l2", {}),
    )


@router.delete(
    "/cache",
    summary="Clear all cache entries",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_cache(request: Request) -> None:
    pipeline = _get_pipeline(request)
    await pipeline._cache.clear()


@router.delete(
    "/cache/{key}",
    summary="Delete a single cache entry by key",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_cache_entry(key: str, request: Request) -> None:
    pipeline = _get_pipeline(request)
    await pipeline._cache.delete(key)


@router.post(
    "/cache/preload",
    response_model=PreloadResponse,
    summary="Preload frequently-used phrases into the cache",
)
async def preload_phrases(body: PreloadRequest, request: Request) -> PreloadResponse:
    pipeline = _get_pipeline(request)
    phrases = body.phrases or None  # None → use built-in defaults
    try:
        loaded = await pipeline.preload_phrases(
            phrases=phrases,
            voice_id=body.voice_id,
            fmt=body.format,
            sample_rate=body.sample_rate,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total = len(body.phrases) if body.phrases else 10  # 10 = len(_PRELOAD_PHRASES_RU)
    return PreloadResponse(loaded=loaded, total=total, voice_id=body.voice_id)
