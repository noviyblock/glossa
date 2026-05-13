"""Async inference queue with backpressure and per-session result routing.

Architecture
------------
One shared InferenceQueue serves all WebSocket sessions.
Each session submits InferenceTask objects into the queue.
N worker coroutines drain the queue and run ONNX predictions.
Results are routed back via per-session asyncio.Queue (output_queue).

Backpressure
------------
When the queue is full, submit() returns False.
The caller (WebSocket handler) must send a BACKPRESSURE frame and
optionally drop the pending window.

Metrics are reported for:
- queue depth (gauge)
- task wait time (histogram)
- inference time (histogram)
- dropped tasks (counter)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable
from uuid import UUID

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY
from prometheus_client import Counter, Gauge, Histogram

from ..domain.pipeline import GestureClassifier
from ..domain.sliding_window import WindowReady

logger = get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
_QUEUE_DEPTH = Gauge(
    "glossa_cv_inference_queue_depth",
    "Number of tasks waiting in the inference queue",
)
_QUEUE_WAIT = Histogram(
    "glossa_cv_inference_queue_wait_seconds",
    "Time a task spent waiting in the inference queue",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
_INFERENCE_TIME = Histogram(
    "glossa_cv_inference_latency_seconds",
    "ONNX model inference time per window",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
_DROPPED = Counter(
    "glossa_cv_inference_dropped_total",
    "Tasks dropped due to full queue (backpressure)",
    ["reason"],
)
_PREDICTIONS = Counter(
    "glossa_cv_predictions_total",
    "Total predictions emitted",
    ["gloss"],
)


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass(slots=True)
class InferenceTask:
    session_id: str
    window: WindowReady
    output_queue: asyncio.Queue               # per-session result sink
    submitted_at: float = field(default_factory=time.perf_counter)
    trace_id: str = ""


@dataclass(slots=True)
class InferenceResult:
    session_id: str
    labels: list[str]
    confidences: list[float]
    start_seq: int
    end_seq: int
    inference_ms: float
    queue_wait_ms: float


# ── Queue ─────────────────────────────────────────────────────────────────────
class InferenceQueue:
    """Bounded async inference queue with N worker coroutines.

    Parameters
    ----------
    classifier:
        ONNX/OpenVINO GestureClassifier instance.
    maxsize:
        Hard cap on queued tasks.  submit() returns False when exceeded.
    num_workers:
        Parallel worker coroutines.  For CPU-bound ONNX, 1 is usually
        best; increase only for GPU where batching helps.
    """

    def __init__(
        self,
        classifier: GestureClassifier,
        maxsize: int = 32,
        num_workers: int = 1,
    ) -> None:
        self._classifier = classifier
        self._queue: asyncio.Queue[InferenceTask | None] = asyncio.Queue(maxsize=maxsize)
        self._num_workers = num_workers
        self._workers: list[asyncio.Task] = []
        self._maxsize = maxsize

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        for i in range(self._num_workers):
            task = asyncio.create_task(self._worker_loop(worker_id=i), name=f"inference-worker-{i}")
            self._workers.append(task)
        logger.info("inference_queue_started", workers=self._num_workers, maxsize=self._maxsize)

    async def stop(self) -> None:
        for _ in self._workers:
            await self._queue.put(None)    # sentinel per worker
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("inference_queue_stopped")

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(self, task: InferenceTask) -> bool:
        """Non-blocking submit.  Returns False (backpressure) if queue full."""
        try:
            self._queue.put_nowait(task)
            _QUEUE_DEPTH.set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            _DROPPED.labels(reason="queue_full").inc()
            logger.warning(
                "inference_queue_full",
                session_id=task.session_id,
                queue_depth=self._queue.qsize(),
            )
            return False

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def utilization(self) -> float:
        return self._queue.qsize() / self._maxsize if self._maxsize else 0.0

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        logger.debug("inference_worker_started", worker_id=worker_id)
        while True:
            task = await self._queue.get()
            _QUEUE_DEPTH.set(self._queue.qsize())

            if task is None:           # shutdown sentinel
                self._queue.task_done()
                break

            try:
                await self._process(task)
            except Exception:
                logger.exception(
                    "inference_worker_error",
                    session_id=task.session_id,
                    worker_id=worker_id,
                )
            finally:
                self._queue.task_done()

        logger.debug("inference_worker_stopped", worker_id=worker_id)

    async def _process(self, task: InferenceTask) -> None:
        wait_ms = (time.perf_counter() - task.submitted_at) * 1000
        _QUEUE_WAIT.observe(wait_ms / 1000)

        t_inf = time.perf_counter()
        with _INFERENCE_TIME.time():
            labels, confidences = await self._classifier.predict(task.window.feature_matrix)
        inference_ms = (time.perf_counter() - t_inf) * 1000

        for label in labels:
            _PREDICTIONS.labels(gloss=label).inc()

        result = InferenceResult(
            session_id=task.session_id,
            labels=labels,
            confidences=confidences,
            start_seq=task.window.start_seq,
            end_seq=task.window.end_seq,
            inference_ms=round(inference_ms, 2),
            queue_wait_ms=round(wait_ms, 2),
        )

        try:
            task.output_queue.put_nowait(result)
        except asyncio.QueueFull:
            _DROPPED.labels(reason="output_queue_full").inc()
            logger.warning(
                "output_queue_full_dropping_result",
                session_id=task.session_id,
            )

        logger.debug(
            "inference_done",
            session_id=task.session_id,
            labels=labels,
            inference_ms=round(inference_ms, 2),
            queue_wait_ms=round(wait_ms, 2),
        )
