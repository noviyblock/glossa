import time

from fastapi import APIRouter, Request, status

from glossa_common.schemas.base import ComponentHealth, HealthResponse

router = APIRouter()
_START = time.time()
_VERSION = "0.1.0"


@router.get("/health/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness() -> HealthResponse:
    return HealthResponse(service="max-adapter", status="healthy", version=_VERSION,
                          uptime_seconds=round(time.time() - _START, 2))


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    client = getattr(request.app.state, "client", None)
    healthy = client is not None and await client.health_check()
    status_str = "healthy" if healthy else "degraded"
    return HealthResponse(
        service="max-adapter", status=status_str, version=_VERSION,
        components=[ComponentHealth(name="max-sdk", status=status_str,
                                    message="fallback mode" if client and not client._use_max else None)],
        uptime_seconds=round(time.time() - _START, 2),
    )
