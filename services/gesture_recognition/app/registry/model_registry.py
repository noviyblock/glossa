"""Model registry — lifecycle management with atomic hot-reload.

Responsibilities:
  - Keep a dict of loaded InferenceEnginePort instances
  - Track which model is "active" for inference
  - Atomic hot-reload: load new → warmup → swap → unload old
  - Thread-safe via asyncio.Lock

Hot-reload protocol
-------------------
1. Load new engine from new_spec (in parallel to serving)
2. Run warmup on new engine
3. Acquire swap lock (brief)
4. Swap self._active to new engine
5. Release lock
6. Unload old engine asynchronously

Model IDs are arbitrary slugs.  The registry doesn't care — callers own naming.
"""

from __future__ import annotations

import asyncio
import time

from glossa_common.logging import get_logger

from ..domain.entities import ModelSpec, ModelStatus, WarmupResult
from ..domain.interfaces import InferenceEnginePort, ModelRegistryPort
from ..engine.factory import create_engine

logger = get_logger(__name__)


class ModelNotFoundError(KeyError):
    pass


class ModelAlreadyLoadedError(RuntimeError):
    pass


class ModelRegistry(ModelRegistryPort):
    """In-process model registry.

    Thread-safety: relies on the asyncio event loop (single-threaded coroutines).
    `_swap_lock` serializes hot-reload transitions.
    """

    def __init__(
        self,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
        warmup_iterations: int = 5,
    ) -> None:
        self._labels = labels
        self._feature_dim = feature_dim
        self._n_threads = n_threads
        self._warmup_iters = warmup_iterations

        self._engines: dict[str, InferenceEnginePort] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._active_id: str | None = None
        self._swap_lock = asyncio.Lock()

    # ── ModelRegistryPort ─────────────────────────────────────────────────────

    async def load(self, spec: ModelSpec) -> None:
        if spec.id in self._engines:
            raise ModelAlreadyLoadedError(f"Model '{spec.id}' is already loaded")

        logger.info("registry_loading_model", model_id=spec.id, path=str(spec.path))
        spec.status = ModelStatus.LOADING

        try:
            engine = await create_engine(
                spec,
                self._labels,
                self._feature_dim,
                self._n_threads,
            )
        except Exception as exc:
            spec.mark_failed(str(exc))
            logger.exception("registry_load_failed", model_id=spec.id)
            raise

        warmup = await engine.warmup(self._warmup_iters)
        if not warmup.success:
            spec.mark_failed(f"Warmup failed: {warmup.error}")
            raise RuntimeError(f"Model warmup failed for {spec.id}: {warmup.error}")

        spec.mark_loaded()
        self._engines[spec.id] = engine
        self._specs[spec.id] = spec

        # Auto-set as active if it's the first model
        if self._active_id is None:
            self._active_id = spec.id

        logger.info(
            "registry_model_loaded",
            model_id=spec.id,
            warmup_mean_ms=round(warmup.mean_ms, 2),
            is_active=self._active_id == spec.id,
        )

    async def unload(self, model_id: str) -> None:
        if model_id not in self._engines:
            raise ModelNotFoundError(model_id)

        if model_id == self._active_id:
            raise RuntimeError("Cannot unload the active model — set another as active first")

        del self._engines[model_id]
        if model_id in self._specs:
            self._specs[model_id].status = ModelStatus.UNLOADED
            del self._specs[model_id]

        logger.info("registry_model_unloaded", model_id=model_id)

    async def hot_reload(self, model_id: str, new_spec: ModelSpec) -> None:
        """Atomically swap a loaded model with a new version.

        The old engine keeps serving until the new one is ready.
        """
        if model_id not in self._engines:
            raise ModelNotFoundError(model_id)

        logger.info("registry_hot_reload_start", model_id=model_id, new_version=new_spec.version)
        old_spec = self._specs[model_id]
        old_spec.status = ModelStatus.HOT_RELOADING

        # Load new engine outside the swap lock (may take seconds)
        try:
            new_engine = await create_engine(
                new_spec, self._labels, self._feature_dim, self._n_threads
            )
            warmup = await new_engine.warmup(self._warmup_iters)
            if not warmup.success:
                raise RuntimeError(f"Warmup failed: {warmup.error}")
            new_spec.mark_loaded()
        except Exception as exc:
            old_spec.status = ModelStatus.LOADED
            logger.exception("registry_hot_reload_failed", model_id=model_id)
            raise

        # Atomic swap
        async with self._swap_lock:
            old_engine = self._engines[model_id]
            self._engines[model_id] = new_engine
            self._specs[model_id] = new_spec
            if self._active_id == model_id:
                pass  # stays active, just pointing at new engine

        # Cleanup old engine (best-effort, log errors but don't raise)
        try:
            del old_engine
        except Exception:
            pass

        logger.info(
            "registry_hot_reload_complete",
            model_id=model_id,
            new_version=new_spec.version,
            warmup_ms=round(warmup.mean_ms, 2),
        )

    def get_active(self) -> InferenceEnginePort:
        if self._active_id is None:
            raise RuntimeError("No active model — load at least one model first")
        return self._engines[self._active_id]

    def set_active(self, model_id: str) -> None:
        if model_id not in self._engines:
            raise ModelNotFoundError(model_id)
        old = self._active_id
        self._active_id = model_id
        logger.info("registry_active_model_changed", from_id=old, to_id=model_id)

    def get(self, model_id: str) -> InferenceEnginePort:
        try:
            return self._engines[model_id]
        except KeyError:
            raise ModelNotFoundError(model_id)

    def list_all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    # ── Extras ────────────────────────────────────────────────────────────────

    @property
    def active_model_id(self) -> str | None:
        return self._active_id

    @property
    def loaded_count(self) -> int:
        return len(self._engines)

    def has_model(self, model_id: str) -> bool:
        return model_id in self._engines

    async def load_many(self, specs: list[ModelSpec]) -> None:
        """Load multiple models concurrently."""
        await asyncio.gather(*[self.load(s) for s in specs], return_exceptions=False)
