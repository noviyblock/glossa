import time

from fastapi import APIRouter, Request, status

from glossa_common.schemas.base import ComponentHealth, HealthResponse

router = APIRouter()
_START = time.time()
_VERSION = "0.1.0"


@router.get("/health/live", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def liveness() -> HealthResponse:
    return HealthResponse(service="nlp-service", status="healthy", version=_VERSION,
                          uptime_seconds=round(time.time() - _START, 2))


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    runner = getattr(request.app.state, "runner", None)
    retriever = getattr(request.app.state, "retriever", None)
    model_ok = runner is not None and runner._model is not None
    rag_ok = retriever is not None and retriever._client is not None
    components = [
        ComponentHealth(name="qwen2-model", status="healthy" if model_ok else "degraded"),
        ComponentHealth(name="qdrant-rag", status="healthy" if rag_ok else "degraded"),
    ]
    overall = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
    return HealthResponse(service="nlp-service", status=overall, version=_VERSION,
                          components=components, uptime_seconds=round(time.time() - _START, 2))
