import time

from fastapi import APIRouter, Request, status

from glossa_common.schemas.base import ComponentHealth, HealthResponse

from ...config import get_settings

router = APIRouter()
_START_TIME = time.time()
_VERSION = "0.1.0"


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    status_code=status.HTTP_200_OK,
)
async def liveness() -> HealthResponse:
    return HealthResponse(
        service="api-gateway",
        status="healthy",
        version=_VERSION,
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe — checks all dependencies",
)
async def readiness(request: Request) -> HealthResponse:
    components: list[ComponentHealth] = []
    overall = "healthy"

    # Redis check
    try:
        t0 = time.perf_counter()
        await request.app.state.redis.ping()
        latency = (time.perf_counter() - t0) * 1000
        components.append(ComponentHealth(name="redis", status="healthy", latency_ms=round(latency, 2)))
    except Exception as exc:
        components.append(ComponentHealth(name="redis", status="unhealthy", message=str(exc)))
        overall = "unhealthy"

    settings = get_settings()
    downstream = {
        "cv-service": settings.cv_service_url,
        "asr-service": settings.asr_service_url,
        "nlp-service": settings.nlp_service_url,
        "tts-service": settings.tts_service_url,
    }

    for svc_name, base_url in downstream.items():
        try:
            t0 = time.perf_counter()
            resp = await request.app.state.http_client.get(
                f"{base_url}/health/live", timeout=3.0
            )
            latency = (time.perf_counter() - t0) * 1000
            svc_status = "healthy" if resp.status_code == 200 else "degraded"
            components.append(
                ComponentHealth(name=svc_name, status=svc_status, latency_ms=round(latency, 2))
            )
            if svc_status != "healthy" and overall == "healthy":
                overall = "degraded"
        except Exception as exc:
            components.append(ComponentHealth(name=svc_name, status="unhealthy", message=str(exc)))
            overall = "degraded"

    return HealthResponse(
        service="api-gateway",
        status=overall,
        version=_VERSION,
        components=components,
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )
