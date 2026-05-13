"""Shared pytest fixtures for gesture_recognition tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain.entities import (
    GlossCandidate,
    InferenceDevice,
    ModelArchitecture,
    ModelBackend,
    ModelSpec,
    ModelStatus,
    QuantizationType,
    TopKResult,
    WarmupResult,
)
from app.domain.interfaces import InferenceEnginePort


# ── Fake labels ───────────────────────────────────────────────────────────────

FAKE_LABELS = [
    "ЗДРАВСТВУЙТЕ", "Я", "ВЫ", "НЕТ", "ДА",
    "СПАСИБО", "ПОЖАЛУЙСТА", "КАК", "ХОРОШО", "ПОМОГИТЕ",
]


# ── Mock engine ───────────────────────────────────────────────────────────────

class MockInferenceEngine(InferenceEnginePort):
    def __init__(
        self,
        labels: list[str] = None,
        model_id: str = "mock-model",
        latency_ms: float = 5.0,
        top_label: str = "ЗДРАВСТВУЙТЕ",
        top_confidence: float = 0.85,
    ) -> None:
        self._labels = labels or FAKE_LABELS
        self._model_id = model_id
        self._latency_ms = latency_ms
        self._top_label = top_label
        self._top_confidence = top_confidence
        self.infer_call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def labels(self) -> list[str]:
        return self._labels

    async def infer(self, feature_sequence: np.ndarray) -> TopKResult:
        self.infer_call_count += 1
        await asyncio.sleep(self._latency_ms / 1000.0)
        idx = self._labels.index(self._top_label) if self._top_label in self._labels else 0
        return TopKResult(
            candidates=[
                GlossCandidate(
                    confidence=self._top_confidence,
                    gloss=self._top_label,
                    gloss_id=idx,
                )
            ],
        )

    async def infer_batch(self, feature_matrix: np.ndarray) -> list[np.ndarray]:
        b = feature_matrix.shape[0]
        logits = np.zeros((b, len(self._labels)), dtype=np.float32)
        idx = self._labels.index(self._top_label) if self._top_label in self._labels else 0
        logits[:, idx] = 5.0  # high logit for top label
        return [logits[i] for i in range(b)]

    async def warmup(self, iterations: int = 5) -> WarmupResult:
        return WarmupResult(
            model_id=self._model_id,
            iterations=iterations,
            total_ms=self._latency_ms * iterations,
            mean_ms=self._latency_ms,
            success=True,
        )

    async def unload(self) -> None:
        pass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def labels() -> list[str]:
    return FAKE_LABELS.copy()


@pytest.fixture
def mock_engine(labels) -> MockInferenceEngine:
    return MockInferenceEngine(labels=labels)


@pytest.fixture
def fake_spec(tmp_path) -> ModelSpec:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"FAKE_ONNX")
    labels_file = tmp_path / "labels.txt"
    labels_file.write_text("\n".join(FAKE_LABELS), encoding="utf-8")
    return ModelSpec(
        id="mock-model",
        name="Mock STGCN",
        architecture=ModelArchitecture.STGCN,
        backend=ModelBackend.ONNX,
        quantization=QuantizationType.FLOAT32,
        path=model_file,
        labels_path=labels_file,
        version="1.0.0",
    )


@pytest.fixture
def feature_sequence() -> np.ndarray:
    """Synthetic [30, 258] feature sequence."""
    return np.random.randn(30, 258).astype(np.float32)
