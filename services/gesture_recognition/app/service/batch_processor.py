"""Dynamic batch processor — collects InferenceRequests and runs them together."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import (
    BatchInferenceRequest,
    GlossCandidate,
    InferenceRequest,
    InferenceResult,
    TopKResult,
)
from ..domain.interfaces import InferenceEnginePort

logger = get_logger(__name__)


class BatchProcessor:
    """Accumulates InferenceRequests and dispatches them as batches.

    Two flush triggers:
    - max_batch_size: flush immediately when the batch is full
    - timeout_ms: flush whatever is queued after this many milliseconds

    The processor runs a background task; callers submit requests and
    receive results via per-request asyncio.Future objects.
    """

    def __init__(
        self,
        engine: InferenceEnginePort,
        max_batch_size: int = 8,
        timeout_ms: float = 20.0,
    ) -> None:
        self._engine = engine
        self._max_batch = max_batch_size
        self._timeout = timeout_ms / 1000.0

        self._queue: asyncio.Queue[tuple[InferenceRequest, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="batch-processor")
        logger.info(
            "batch_processor_started",
            max_batch=self._max_batch,
            timeout_ms=self._timeout * 1000,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("batch_processor_stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit(self, request: InferenceRequest) -> InferenceResult:
        """Submit a single request; await its result."""
        future: asyncio.Future[InferenceResult] = asyncio.get_event_loop().create_future()
        await self._queue.put((request, future))
        return await future

    # ── Background loop ───────────────────────────────────────────────────────

    async def _run(self) -> None:
        while self._running:
            batch: list[tuple[InferenceRequest, asyncio.Future]] = []
            deadline = time.monotonic() + self._timeout

            # Collect up to max_batch_size items within the timeout window
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            await self._dispatch(batch)

    async def _dispatch(
        self,
        batch: list[tuple[InferenceRequest, asyncio.Future]],
    ) -> None:
        requests = [req for req, _ in batch]
        futures = [fut for _, fut in batch]

        batch_req = BatchInferenceRequest(requests=requests)
        t0 = time.perf_counter()

        try:
            # Stack into [B, T, D]
            feature_matrix = batch_req.feature_matrix()
            raw_outputs = await self._engine.infer_batch(feature_matrix)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "batch_dispatched",
                batch_size=len(requests),
                inference_ms=round(elapsed_ms, 2),
            )

            for i, (req, fut) in enumerate(zip(requests, futures)):
                if fut.done():
                    continue
                try:
                    result = self._build_result(req, raw_outputs[i], elapsed_ms)
                    fut.set_result(result)
                except Exception as exc:
                    fut.set_exception(exc)

        except Exception as exc:
            logger.exception("batch_dispatch_failed", batch_size=len(requests))
            for _, fut in zip(requests, futures):
                if not fut.done():
                    fut.set_exception(exc)

    def _build_result(
        self,
        req: InferenceRequest,
        logits: np.ndarray,
        inference_ms: float,
    ) -> InferenceResult:
        from ..engine.base import _softmax, _topk  # local import avoids circular

        probs = _softmax(logits)
        top_indices, top_probs = _topk(probs, req.top_k)

        candidates = [
            GlossCandidate(
                confidence=float(top_probs[j]),
                gloss=self._engine.labels[top_indices[j]],
                gloss_id=int(top_indices[j]),
            )
            for j in range(len(top_indices))
            if top_probs[j] >= req.confidence_threshold
        ]

        window = TopKResult(
            candidates=candidates,
            window_start=0,
            window_end=0,
        )

        from ..domain.entities import InferenceDevice
        return InferenceResult(
            request_id=req.request_id,
            session_id=req.session_id,
            top_k=[window],
            model_id=self._engine.model_id,
            model_version=self._engine.model_version,
            inference_ms=inference_ms,
            queue_wait_ms=(time.perf_counter() - req.submitted_at) * 1000.0 - inference_ms,
            batch_size=1,
            device=InferenceDevice.CPU,
        )
