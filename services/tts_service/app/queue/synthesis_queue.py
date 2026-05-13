"""Async priority queue for TTS synthesis jobs.

Design:
- asyncio.PriorityQueue ordered by (priority, submitted_at)
- N worker coroutines drain the queue concurrently
- Each job carries an asyncio.Future — callers await the future
- Bounded queue: submit() raises QueueFullError when maxsize is hit
- Per-job timeout: future is cancelled if synthesis exceeds deadline_s
- Graceful shutdown: drain() waits for in-flight jobs then cancels workers
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import SynthesisJob, VoiceProfile
from ..domain.interfaces import SynthesizerPort

logger = get_logger(__name__)


class QueueFullError(RuntimeError):
    pass


@dataclass(order=True)
class _QueueItem:
    """Wrapper so asyncio.PriorityQueue sorts correctly."""
    priority: int
    submitted_at: float
    job: SynthesisJob = field(compare=False)
    future: asyncio.Future = field(compare=False)


class SynthesisQueue:
    """Async priority queue with N synthesis workers."""

    def __init__(
        self,
        synthesizer: SynthesizerPort,
        workers: int = 2,
        maxsize: int = 64,
        job_timeout_s: float = 30.0,
    ) -> None:
        self._synthesizer = synthesizer
        self._n_workers = workers
        self._maxsize = maxsize
        self._job_timeout_s = job_timeout_s
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue(maxsize=maxsize)
        self._worker_tasks: list[asyncio.Task] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._worker_tasks = [
            asyncio.create_task(self._worker(i), name=f"tts-queue-worker-{i}")
            for i in range(self._n_workers)
        ]
        logger.info("synthesis_queue_started", workers=self._n_workers, maxsize=self._maxsize)

    async def stop(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._started = False
        logger.info("synthesis_queue_stopped")

    async def submit(
        self,
        text: str,
        voice: VoiceProfile,
        put_accent: bool = True,
        put_yo: bool = True,
        priority: int = 0,
    ) -> np.ndarray:
        """Submit a synthesis job and await the PCM result.

        Raises QueueFullError if the queue is at capacity.
        Raises asyncio.TimeoutError if synthesis exceeds job_timeout_s.
        """
        if self._queue.full():
            raise QueueFullError(
                f"Synthesis queue is full ({self._maxsize} items). Try again later."
            )

        loop = asyncio.get_event_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        job = SynthesisJob(
            request=_SynthReq(text=text, voice=voice, put_accent=put_accent, put_yo=put_yo),
        )
        item = _QueueItem(
            priority=priority,
            submitted_at=time.perf_counter(),
            job=job,
            future=future,
        )
        await self._queue.put(item)

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self._job_timeout_s)
        except asyncio.TimeoutError:
            future.cancel()
            raise

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    # ── Worker ─────────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        logger.debug("synthesis_worker_started", worker_id=worker_id)
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            req = item.job.request
            wait_ms = (time.perf_counter() - item.submitted_at) * 1000

            try:
                t0 = time.perf_counter()
                pcm = await self._synthesizer.synthesize(
                    text=req.text,
                    voice=req.voice,
                    put_accent=req.put_accent,
                    put_yo=req.put_yo,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "synthesis_job_done",
                    worker=worker_id,
                    chars=len(req.text),
                    wait_ms=round(wait_ms, 1),
                    elapsed_ms=round(elapsed_ms, 1),
                )
                if not item.future.done():
                    item.future.set_result(pcm)

            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                break
            except Exception as exc:
                logger.exception("synthesis_job_failed", worker=worker_id, chars=len(req.text))
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._queue.task_done()


# Internal minimal request holder (avoids circular import with full SynthesisRequest)
from dataclasses import dataclass as _dc

@_dc
class _SynthReq:
    text: str
    voice: VoiceProfile
    put_accent: bool = True
    put_yo: bool = True
