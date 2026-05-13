"""Domain layer — MAX adapter pipeline orchestration."""
from __future__ import annotations

import time
from uuid import UUID

from glossa_common.logging import get_logger

logger = get_logger(__name__)


class MAXPipelineRequest:
    def __init__(
        self,
        session_id: UUID,
        model_name: str,
        inputs: dict[str, object],
        timeout: float = 30.0,
    ) -> None:
        self.session_id = session_id
        self.model_name = model_name
        self.inputs = inputs
        self.timeout = timeout


class MAXPipelineResult:
    def __init__(
        self,
        session_id: UUID,
        model_name: str,
        outputs: dict[str, object],
        latency_ms: float,
        backend: str,
    ) -> None:
        self.session_id = session_id
        self.model_name = model_name
        self.outputs = outputs
        self.latency_ms = latency_ms
        self.backend = backend

    def to_dict(self) -> dict:
        return {
            "session_id": str(self.session_id),
            "model_name": self.model_name,
            "outputs": self.outputs,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
        }


class MAXAdapterDomain:
    def __init__(self, client: object) -> None:
        self._client = client

    async def run(self, request: MAXPipelineRequest) -> MAXPipelineResult:
        from ..infrastructure.max_client import MAXPipelineClient

        client: MAXPipelineClient = self._client  # type: ignore[assignment]
        t0 = time.perf_counter()

        outputs = await client.run_inference(request.model_name, request.inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "max_pipeline_completed",
            session=str(request.session_id),
            model=request.model_name,
            latency_ms=round(elapsed_ms, 2),
        )

        return MAXPipelineResult(
            session_id=request.session_id,
            model_name=request.model_name,
            outputs=outputs,
            latency_ms=round(elapsed_ms, 2),
            backend="max" if client._use_max else "fallback",
        )
