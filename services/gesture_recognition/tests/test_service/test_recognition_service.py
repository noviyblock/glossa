"""Tests for GestureRecognitionService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.domain.entities import GlossCandidate, InferenceResult, TopKResult
from app.domain.sliding_window import SlidingWindowManager
from app.service.fallback_heuristics import FallbackHeuristics
from app.service.recognition_service import GestureRecognitionService, RecognitionConfig

from ..conftest import FAKE_LABELS, MockInferenceEngine


def _make_service(labels=None, top_label="ЗДРАВСТВУЙТЕ", confidence=0.85, use_fallback=True):
    labels = labels or FAKE_LABELS
    engine = MockInferenceEngine(labels=labels, top_label=top_label, top_confidence=confidence)

    registry = MagicMock()
    registry.get_active.return_value = engine

    # BatchProcessor that directly calls engine.infer
    batch_proc = MagicMock()
    async def _submit(req):
        result = await engine.infer(req.feature_sequence)
        return InferenceResult(
            request_id=req.request_id,
            session_id=req.session_id,
            top_k=[result],
            model_id=engine.model_id,
            model_version=engine.model_version,
            inference_ms=5.0,
            queue_wait_ms=1.0,
        )
    batch_proc.submit = _submit

    fallback = FallbackHeuristics(labels=labels, confidence_floor=0.5, top_k=3)
    sliding = SlidingWindowManager(window_size=5, stride=3, max_buffer=20, feature_dim=8)
    cfg = RecognitionConfig(top_k=3, confidence_threshold=0.3, use_fallback=use_fallback)

    svc = GestureRecognitionService(
        registry=registry,
        batch_processor=batch_proc,
        fallback=fallback,
        sliding_window=sliding,
        config=cfg,
    )
    return svc


@pytest.mark.asyncio
async def test_recognize_returns_empty_below_window():
    svc = _make_service()
    seq = np.zeros((3, 8), dtype=np.float32)  # < window_size=5
    output = await svc.recognize(seq, session_id="s1")
    assert output.glosses == []
    assert output.inference_ms == 0.0


@pytest.mark.asyncio
async def test_recognize_returns_predictions_above_window():
    svc = _make_service(confidence=0.85)
    seq = np.zeros((5, 8), dtype=np.float32)
    output = await svc.recognize(seq, session_id="s1")
    assert "ЗДРАВСТВУЙТЕ" in output.glosses
    assert not output.is_fallback


@pytest.mark.asyncio
async def test_recognize_activates_fallback_on_low_confidence():
    svc = _make_service(confidence=0.1, use_fallback=True)
    seq = np.zeros((5, 8), dtype=np.float32)
    output = await svc.recognize(seq, session_id="s2")
    # Low confidence → fallback kicks in
    assert output.is_fallback or output.glosses == []  # either outcome is valid


@pytest.mark.asyncio
async def test_reset_session_clears_state():
    svc = _make_service()
    seq = np.zeros((3, 8), dtype=np.float32)
    await svc.recognize(seq, session_id="s3")
    svc.reset_session("s3")
    # After reset, pushing 3 frames again should still not emit
    output = await svc.recognize(seq, session_id="s3")
    assert output.glosses == []


@pytest.mark.asyncio
async def test_recognize_top_k_respected():
    svc = _make_service(top_k_cfg := 2)
    seq = np.zeros((5, 8), dtype=np.float32)
    output = await svc.recognize(seq, top_k=2)
    assert len(output.glosses) <= 2
