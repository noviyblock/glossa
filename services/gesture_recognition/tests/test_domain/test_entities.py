"""Tests for domain entities."""

import numpy as np
import pytest

from app.domain.entities import (
    GlossCandidate,
    LatencyStats,
    ModelStatus,
    TopKResult,
)


def test_gloss_candidate_confidence_bounds():
    with pytest.raises(ValueError):
        GlossCandidate(confidence=1.1, gloss="TEST", gloss_id=0)
    with pytest.raises(ValueError):
        GlossCandidate(confidence=-0.1, gloss="TEST", gloss_id=0)


def test_gloss_candidate_valid():
    c = GlossCandidate(confidence=0.75, gloss="ЗДРАВСТВУЙТЕ", gloss_id=0)
    assert c.confidence == 0.75


def test_top_k_result_best():
    candidates = [
        GlossCandidate(confidence=0.9, gloss="A", gloss_id=0),
        GlossCandidate(confidence=0.5, gloss="B", gloss_id=1),
    ]
    result = TopKResult(candidates=candidates)
    assert result.best.gloss == "A"
    assert result.best_confidence == 0.9


def test_top_k_result_empty():
    result = TopKResult(candidates=[])
    assert result.best is None
    assert result.best_gloss == "<unknown>"
    assert result.best_confidence == 0.0


def test_latency_stats_from_samples():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = LatencyStats.from_samples(samples)
    assert stats.mean_ms == pytest.approx(30.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(50.0)
    assert stats.p50_ms == pytest.approx(30.0)
    assert stats.p95_ms <= stats.max_ms


def test_model_spec_mark_loaded(fake_spec):
    fake_spec.mark_loaded()
    assert fake_spec.status == ModelStatus.LOADED
    assert fake_spec.loaded_at is not None
    assert fake_spec.load_error is None


def test_model_spec_mark_failed(fake_spec):
    fake_spec.mark_failed("onnx session creation failed")
    assert fake_spec.status == ModelStatus.FAILED
    assert "onnx" in fake_spec.load_error
