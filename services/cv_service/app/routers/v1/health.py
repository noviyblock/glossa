import time

from fastapi import APIRouter, Request, status

from glossa_common.schemas.base import ComponentHealth, HealthResponse

router = APIRouter()
_START = time.time()
_VERSION = "0.1.0"


@router.get("/health/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness() -> HealthResponse:
    return HealthResponse(service="cv-service", status="healthy", version=_VERSION,
                          uptime_seconds=round(time.time() - _START, 2))


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    components: list[ComponentHealth] = []
    pipeline = getattr(request.app.state, "pipeline", None)
    mediapipe = getattr(request.app.state, "mediapipe", None)
    components.append(ComponentHealth(
        name="cv-pipeline", status="healthy" if pipeline else "unhealthy"
    ))
    components.append(ComponentHealth(
        name="mediapipe", status="healthy" if mediapipe else "unhealthy"
    ))
    overall = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
    return HealthResponse(service="cv-service", status=overall, version=_VERSION,
                          components=components, uptime_seconds=round(time.time() - _START, 2))
