"""Integration test for InferenceQueue with a mock classifier."""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.domain.pipeline import GestureClassifier
from app.domain.sliding_window import FrameMeta, SlidingWindowBuffer, WindowReady
from app.inference.queue import InferenceQueue, InferenceResult, InferenceTask


class MockClassifier(GestureClassifier):
    def __init__(self, label: str = "ВРАЧ", confidence: float = 0.92) -> None:
        self._label = label
        self._confidence = confidence
        self.call_count = 0

    async def predict(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        self.call_count += 1
        await asyncio.sleep(0)  # simulate async
        return [self._label], [self._confidence]

    async def warmup(self) -> None:
        pass


def _make_window(window_size: int = 5) -> WindowReady:
    feat = np.zeros((window_size, 8), dtype=np.float32)
    meta = [FrameMeta(seq=i, ts_ms=float(i * 33), frame_index=i) for i in range(window_size)]
    return WindowReady(
        feature_matrix=feat,
        frames_meta=meta,
        start_seq=0,
        end_seq=window_size - 1,
        buffer_depth=window_size,
    )


@pytest.mark.asyncio
async def test_single_task_processed():
    classifier = MockClassifier()
    queue = InferenceQueue(classifier=classifier, maxsize=8, num_workers=1)
    await queue.start()

    output_q: asyncio.Queue[InferenceResult] = asyncio.Queue()
    task = InferenceTask(session_id="test-session", window=_make_window(), output_queue=output_q)

    accepted = queue.submit(task)
    assert accepted

    result = await asyncio.wait_for(output_q.get(), timeout=2.0)
    assert result.labels == ["ВРАЧ"]
    assert result.confidences[0] == pytest.approx(0.92)
    assert result.session_id == "test-session"
    assert result.inference_ms >= 0
    assert result.queue_wait_ms >= 0

    await queue.stop()
    assert classifier.call_count == 1


@pytest.mark.asyncio
async def test_multiple_tasks_all_processed():
    classifier = MockClassifier()
    queue = InferenceQueue(classifier=classifier, maxsize=16, num_workers=1)
    await queue.start()

    output_q: asyncio.Queue[InferenceResult] = asyncio.Queue()
    n = 5
    for i in range(n):
        task = InferenceTask(
            session_id=f"session-{i}", window=_make_window(), output_queue=output_q
        )
        assert queue.submit(task)

    results = []
    for _ in range(n):
        r = await asyncio.wait_for(output_q.get(), timeout=3.0)
        results.append(r)

    await queue.stop()
    assert len(results) == n
    assert classifier.call_count == n


@pytest.mark.asyncio
async def test_backpressure_when_full():
    classifier = MockClassifier()
    queue = InferenceQueue(classifier=classifier, maxsize=2, num_workers=0)
    # num_workers=0 → no consumers → queue fills up

    output_q: asyncio.Queue = asyncio.Queue()
    task = InferenceTask(session_id="s", window=_make_window(), output_queue=output_q)

    q1 = queue.submit(task)
    q2 = queue.submit(task)
    q3 = queue.submit(task)   # should be rejected

    assert q1 is True
    assert q2 is True
    assert q3 is False         # backpressure

    # Cleanup: restart with worker to drain
    queue._num_workers = 1
    await queue.start()
    await asyncio.sleep(0.1)
    await queue.stop()


@pytest.mark.asyncio
async def test_queue_depth_gauge():
    classifier = MockClassifier()
    queue = InferenceQueue(classifier=classifier, maxsize=8, num_workers=0)

    output_q: asyncio.Queue = asyncio.Queue()
    for _ in range(4):
        queue.submit(InferenceTask(session_id="s", window=_make_window(), output_queue=output_q))

    assert queue.depth == 4
    assert queue.utilization == pytest.approx(0.5)
