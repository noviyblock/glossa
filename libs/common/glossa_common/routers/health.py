"""Shared health router — import and include in every service."""
import time

from fastapi import APIRouter, Request, status

from ..schemas.base import ComponentHealth, HealthResponse

router = APIRouter()


def make_health_router(service_name: str, version: str = "0.1.0") -> APIRouter:
    r = APIRouter()
    start_time = time.time()

    @r.get(
        "/health/live",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        summary="Liveness probe",
    )
    async def liveness() -> HealthResponse:
        return HealthResponse(
            service=service_name,
            status="healthy",
            version=version,
            uptime_seconds=round(time.time() - start_time, 2),
        )

    @r.get(
        "/health/ready",
        response_model=HealthResponse,
        summary="Readiness probe",
    )
    async def readiness(request: Request) -> HealthResponse:
        components: list[ComponentHealth] = []
        overall = "healthy"

        if hasattr(request.app.state, "redis"):
            try:
                t0 = time.perf_counter()
                await request.app.state.redis.ping()
                components.append(
                    ComponentHealth(
                        name="redis",
                        status="healthy",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                )
            except Exception as exc:
                components.append(ComponentHealth(name="redis", status="unhealthy", message=str(exc)))
                overall = "degraded"

        model_attrs = ["runner", "pipeline", "client"]
        for attr in model_attrs:
            if hasattr(request.app.state, attr):
                components.append(ComponentHealth(name="model", status="healthy"))
                break

        return HealthResponse(
            service=service_name,
            status=overall,
            version=version,
            components=components,
            uptime_seconds=round(time.time() - start_time, 2),
        )

    return r
