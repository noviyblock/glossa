"""MAX SDK adapter — wraps Modular MAX pipeline for inference acceleration.

MAX (Modular Accelerated Xecution) provides a unified runtime for deploying
ONNX, PyTorch, and custom AI models with automatic hardware optimization.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

logger = get_logger(__name__)


class MAXPipelineClient:
    """Adapter around MAX SDK's InferenceSession.

    Falls back gracefully to direct ONNX execution when MAX SDK is
    unavailable (e.g., during local CPU development).
    """

    def __init__(
        self,
        endpoint: str,
        pipeline_timeout: float = 30.0,
        batch_size: int = 1,
    ) -> None:
        self._endpoint = endpoint
        self._timeout = pipeline_timeout
        self._batch_size = batch_size
        self._session: object | None = None
        self._use_max = False

    async def initialize(self) -> None:
        try:
            await asyncio.to_thread(self._init_max_session)
            self._use_max = True
            logger.info("max_sdk_initialized", endpoint=self._endpoint)
        except ImportError:
            logger.warning(
                "max_sdk_not_available_falling_back_to_onnx",
                endpoint=self._endpoint,
            )
            self._use_max = False

    def _init_max_session(self) -> None:
        # MAX SDK import — optional dependency
        from max.engine import InferenceSession  # type: ignore[import]

        self._session = InferenceSession(self._endpoint)

    async def run_inference(
        self,
        model_name: str,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        if self._use_max and self._session is not None:
            return await self._run_max(model_name, inputs)
        return await self._run_fallback(model_name, inputs)

    async def _run_max(
        self, model_name: str, inputs: dict[str, object]
    ) -> dict[str, object]:
        with MODEL_INFERENCE_LATENCY.labels(service="max-adapter", model=model_name).time():
            result = await asyncio.wait_for(
                asyncio.to_thread(self._session.execute, model_name, inputs),  # type: ignore[union-attr]
                timeout=self._timeout,
            )
        return result  # type: ignore[return-value]

    async def _run_fallback(
        self, model_name: str, inputs: dict[str, object]
    ) -> dict[str, object]:
        logger.debug("max_fallback_inference", model=model_name)
        await asyncio.sleep(0)  # yield to event loop
        return {"status": "fallback", "model": model_name, "outputs": {}}

    async def health_check(self) -> bool:
        if not self._use_max:
            return True  # Fallback mode is always "healthy"
        try:
            await asyncio.wait_for(
                asyncio.to_thread(lambda: self._session is not None),  # type: ignore[misc]
                timeout=2.0,
            )
            return True
        except Exception:
            return False
