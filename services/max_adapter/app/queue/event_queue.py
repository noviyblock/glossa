"""Async event queue — webhook returns 200 immediately; events processed in background."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any

from glossa_common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedEvent:
    update_type: str
    payload: dict
    trace_id: str
    enqueued_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class EventQueue:
    def __init__(
        self,
        handler: Callable[[QueuedEvent], Coroutine[Any, Any, None]],
        workers: int = 4,
        maxsize: int = 512,
    ) -> None:
        self._handler = handler
        self._workers = workers
        self._queue: asyncio.Queue[QueuedEvent] = asyncio.Queue(maxsize=maxsize)
        self._tasks: list[asyncio.Task] = []
        self._dropped = 0

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"event-worker-{i}")
            for i in range(self._workers)
        ]
        logger.info("event_queue_started", workers=self._workers, maxsize=self._queue.maxsize)

    async def stop(self) -> None:
        for _ in self._tasks:
            await self._queue.put(_SENTINEL)  # type: ignore[arg-type]
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("event_queue_stopped", dropped_total=self._dropped)

    async def enqueue(self, event: QueuedEvent) -> bool:
        """Return False if queue is full (event dropped)."""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "event_queue_full",
                trace_id=event.trace_id,
                dropped_total=self._dropped,
            )
            return False

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    async def _worker(self, worker_id: int) -> None:
        logger.debug("event_worker_ready", worker_id=worker_id)
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._handler(event)
            except Exception:
                logger.exception(
                    "event_worker_error",
                    worker_id=worker_id,
                    trace_id=event.trace_id,
                )
            finally:
                self._queue.task_done()


# Sentinel value to signal workers to stop
_SENTINEL = object()
