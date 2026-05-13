"""Domain interfaces (ports) — abstract contracts between layers.

Each interface defines WHAT is needed without HOW it's implemented.
Concrete adapters live in engine/, registry/, repository/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from .entities import (
    BatchInferenceRequest,
    BenchmarkResult,
    GlossCandidate,
    InferenceDevice,
    InferenceRequest,
    InferenceResult,
    ModelSpec,
    ModelStatus,
    TopKResult,
    WarmupResult,
)


class InferenceEnginePort(ABC):
    """Contract for any inference engine (ONNX, OpenVINO, TensorRT …)."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def device(self) -> InferenceDevice: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    async def predict(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Run inference on a single [T, D] feature sequence.

        Returns raw logits [num_classes].
        """

    @abstractmethod
    async def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Run inference on a [B, T, D] batch.

        Returns raw logits [B, num_classes].
        """

    @abstractmethod
    async def warmup(self, n_iterations: int = 3) -> WarmupResult:
        """Run N dummy inferences to populate caches."""

    @abstractmethod
    async def benchmark(
        self,
        n_samples: int = 100,
        batch_size: int = 1,
        sequence_length: int = 30,
    ) -> BenchmarkResult:
        """Measure latency and throughput under synthetic load."""

    @abstractmethod
    async def reload(self, new_spec: ModelSpec) -> None:
        """Hot-reload model weights without restarting the engine."""


class ModelRegistryPort(ABC):
    """Contract for model registry — lifecycle management of loaded engines."""

    @abstractmethod
    async def load(self, spec: ModelSpec) -> None:
        """Load a model from spec into the registry."""

    @abstractmethod
    async def unload(self, model_id: str) -> None:
        """Unload a model and release resources."""

    @abstractmethod
    async def hot_reload(self, model_id: str, new_spec: ModelSpec) -> None:
        """Atomically swap running model with a new version."""

    @abstractmethod
    def get_active(self) -> InferenceEnginePort:
        """Return the currently active inference engine."""

    @abstractmethod
    def set_active(self, model_id: str) -> None:
        """Switch which loaded model is used for inference."""

    @abstractmethod
    def get(self, model_id: str) -> InferenceEnginePort:
        """Return a specific engine by ID (may not be active)."""

    @abstractmethod
    def list_all(self) -> list[ModelSpec]:
        """Return specs for all loaded models."""


class ModelRepositoryPort(ABC):
    """Contract for model file / metadata storage."""

    @abstractmethod
    async def list_available(self) -> list[ModelSpec]:
        """Scan storage and return all discoverable model specs."""

    @abstractmethod
    async def get_spec(self, model_id: str) -> ModelSpec:
        """Return spec for a specific model ID."""

    @abstractmethod
    async def load_labels(self, model_id: str) -> list[str]:
        """Return ordered list of class labels (RSL glosses)."""

    @abstractmethod
    async def save_benchmark(self, result: BenchmarkResult) -> None:
        """Persist benchmark results for later comparison."""

    @abstractmethod
    async def watch_changes(self) -> AsyncIterator[ModelSpec]:
        """Yield model specs whenever a model file changes (hot-reload trigger)."""


class GlossaryPort(ABC):
    """Contract for RSL glossary access."""

    @abstractmethod
    async def lookup(self, gloss: str) -> dict[str, Any]:
        """Return glossary entry: translations, phonology, frequency."""

    @abstractmethod
    async def fuzzy_match(self, partial: str, top_k: int = 5) -> list[str]:
        """Find the closest glosses to a partial/misspelled input."""

    @abstractmethod
    def get_all_labels(self) -> list[str]:
        """Return the full ordered list of gloss labels (matches model output)."""

    @abstractmethod
    def frequency_prior(self, gloss: str) -> float:
        """Return normalized frequency [0,1] of this gloss in corpus."""


class FallbackHeuristicsPort(ABC):
    """Contract for rule-based fallback when model confidence is low."""

    @abstractmethod
    def predict(
        self,
        feature_sequence: np.ndarray,
        low_conf_candidates: list[GlossCandidate],
    ) -> list[GlossCandidate]:
        """Return heuristic-based candidates when model is uncertain."""


class MLflowTrackerPort(ABC):
    """Contract for MLflow experiment tracking."""

    @abstractmethod
    async def log_benchmark(self, result: BenchmarkResult, experiment: str) -> str:
        """Log benchmark to MLflow run, return run_id."""

    @abstractmethod
    async def log_model_activation(self, model_id: str, model_version: str) -> None:
        """Log which model is currently active."""

    @abstractmethod
    async def compare_models(
        self, model_a: str, model_b: str, experiment: str
    ) -> dict[str, Any]:
        """Compare benchmark metrics between two model versions."""
