import time

from fastapi import APIRouter, Request, status

from glossa_common.schemas.base import ComponentHealth, HealthResponse

router = APIRouter()
_START = time.time()
_VERSION = "0.1.0"


@router.get("/health/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness() -> HealthResponse:
    return HealthResponse(service="asr-service", status="healthy", version=_VERSION,
                          uptime_seconds=round(time.time() - _START, 2))


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    runner = getattr(request.app.state, "runner", None)
    model_loaded = runner is not None and runner._model is not None
    status_str = "healthy" if model_loaded else "degraded"
    return HealthResponse(
        service="asr-service", status=status_str, version=_VERSION,
        components=[ComponentHealth(name="whisper-model", status=status_str)],
        uptime_seconds=round(time.time() - _START, 2),
    )
