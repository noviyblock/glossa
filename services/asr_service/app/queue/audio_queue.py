"""Async audio chunk queue with backpressure and per-session routing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from glossa_common.logging import get_logger

from ..domain.entities import AudioChunk

logger = get_logger(__name__)


@dataclass
class QueueStats:
    submitted: int = 0
    dropped: int = 0
    processed: int = 0
    peak_depth: int = 0


class AudioChunkQueue:
    """Bounded async queue for audio chunks with backpressure signalling.

    Each session has its own queue slot so one slow session cannot block others.
    A shared background worker pool drains all queues.
    """

    def __init__(self, maxsize: int = 128, num_workers: int = 2) -> None:
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._num_workers = num_workers
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._stats = QueueStats()
        self._handlers: dict[str, asyncio.Callable] = {}

    def register_handler(self, session_id: str, handler) -> None:
        """Register an async handler coroutine for a session's chunks."""
        self._handlers[session_id] = handler

    def unregister_handler(self, session_id: str) -> None:
        self._handlers.pop(session_id, None)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"audio-queue-worker-{i}")
            for i in range(self._num_workers)
        ]
        logger.info("audio_queue_started", workers=self._num_workers, maxsize=self._maxsize)

    async def stop(self) -> None:
        self._running = False
        # Send sentinel for each worker
        for _ in range(self._num_workers):
            await self._queue.put(None)
        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                task.cancel()
        self._tasks = []
        logger.info("audio_queue_stopped", stats=vars(self._stats))

    def submit(self, chunk: AudioChunk) -> bool:
        """Non-blocking submit. Returns False if queue is full (backpressure)."""
        try:
            self._queue.put_nowait(chunk)
            self._stats.submitted += 1
            depth = self._queue.qsize()
            if depth > self._stats.peak_depth:
                self._stats.peak_depth = depth
            return True
        except asyncio.QueueFull:
            self._stats.dropped += 1
            logger.warning(
                "audio_queue_full",
                session_id=chunk.session_id,
                dropped_total=self._stats.dropped,
            )
            return False

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def utilization(self) -> float:
        return self._queue.qsize() / self._maxsize

    @property
    def stats(self) -> QueueStats:
        return self._stats

    async def _worker(self, worker_id: int) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            t0 = time.perf_counter()
            handler = self._handlers.get(chunk.session_id)
            if handler is None:
                logger.debug("audio_queue_no_handler", session_id=chunk.session_id)
                continue
            try:
                await handler(chunk)
                self._stats.processed += 1
                wait_ms = (time.perf_counter() - chunk.received_at) * 1000
                if wait_ms > 500:
                    logger.warning("audio_chunk_queue_lag_high", wait_ms=round(wait_ms, 1))
            except Exception:
                logger.exception("audio_queue_handler_error", session_id=chunk.session_id)
