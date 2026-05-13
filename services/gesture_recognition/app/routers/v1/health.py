"""Health endpoints — liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from glossa_common.schemas.base import ComponentHealth, HealthResponse

from ...dependencies import get_registry

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", components={})


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(registry=Depends(get_registry)) -> JSONResponse:
    components: dict[str, ComponentHealth] = {}
    status = "ok"

    # Check that at least one model is loaded
    loaded = registry.loaded_count
    if loaded == 0:
        components["model_registry"] = ComponentHealth(
            status="degraded", detail="No models loaded"
        )
        status = "degraded"
    else:
        components["model_registry"] = ComponentHealth(
            status="ok",
            detail=f"{loaded} model(s) loaded, active={registry.active_model_id}",
        )

    code = 200 if status == "ok" else 503
    return JSONResponse(
        content=HealthResponse(status=status, components=components).model_dump(),
        status_code=code,
    )
