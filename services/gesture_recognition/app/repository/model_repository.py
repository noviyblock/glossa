"""Filesystem model repository — scanning, metadata persistence, file watching."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from glossa_common.logging import get_logger

from ..domain.entities import ModelSpec, ModelStatus
from ..domain.interfaces import ModelRepositoryPort
from ..registry.model_loader import ModelLoader

logger = get_logger(__name__)

_METADATA_CACHE = "model_metadata_cache.json"


class FilesystemModelRepository(ModelRepositoryPort):
    """Discovers and tracks ONNX models on disk.

    Provides:
    - scan() to enumerate all models under models_root
    - get(model_id) to retrieve a cached spec
    - watch() to detect new/changed model files and call back
    """

    def __init__(
        self,
        models_root: Path,
        labels_fallback: Path | None = None,
        poll_interval_s: float = 30.0,
    ) -> None:
        self._root = models_root
        self._loader = ModelLoader(models_root, labels_fallback)
        self._poll_interval = poll_interval_s

        self._specs: dict[str, ModelSpec] = {}
        self._last_scan: float = 0.0
        self._watch_task: asyncio.Task | None = None
        self._on_change: Callable[[ModelSpec], None] | None = None

    # ── ModelRepositoryPort ───────────────────────────────────────────────────

    def scan(self) -> list[ModelSpec]:
        """Synchronously scan the models directory and refresh internal cache."""
        specs = self._loader.scan_all()
        self._specs = {s.id: s for s in specs}
        self._last_scan = time.time()
        logger.info("model_repo_scanned", n_models=len(specs), root=str(self._root))
        return specs

    def get(self, model_id: str) -> ModelSpec | None:
        return self._specs.get(model_id)

    def list_all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def save_metadata(self, spec: ModelSpec) -> None:
        """Persist spec status/timestamps to a JSON sidecar (best-effort)."""
        cache_path = spec.path.parent / _METADATA_CACHE
        try:
            existing: dict = {}
            if cache_path.exists():
                existing = json.loads(cache_path.read_text(encoding="utf-8"))
            existing[spec.id] = {
                "status": spec.status,
                "loaded_at": spec.loaded_at,
                "load_error": spec.load_error,
                "version": spec.version,
            }
            cache_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("model_repo_save_metadata_failed", model_id=spec.id)

    # ── File watcher ──────────────────────────────────────────────────────────

    async def start_watching(self, on_change: Callable[[ModelSpec], None]) -> None:
        """Start polling models_root for new/changed model directories."""
        self._on_change = on_change
        self._watch_task = asyncio.create_task(self._poll_loop(), name="model-repo-watcher")
        logger.info("model_repo_watcher_started", interval_s=self._poll_interval)

    async def stop_watching(self) -> None:
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
        logger.info("model_repo_watcher_stopped")

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._check_for_changes()
            except Exception:
                logger.exception("model_repo_watcher_error")

    async def _check_for_changes(self) -> None:
        if not self._root.exists():
            return

        for candidate in self._root.iterdir():
            if not candidate.is_dir():
                continue

            model_id = candidate.name
            onnx_files = list(candidate.glob("*.onnx"))
            if not onnx_files:
                continue

            latest_mtime = max(f.stat().st_mtime for f in onnx_files)
            existing = self._specs.get(model_id)

            if existing is None or latest_mtime > self._last_scan:
                try:
                    new_spec = self._loader.load_spec(candidate)
                    self._specs[new_spec.id] = new_spec
                    if self._on_change:
                        self._on_change(new_spec)
                    logger.info(
                        "model_repo_new_model_detected",
                        model_id=new_spec.id,
                        path=str(new_spec.path),
                    )
                except Exception:
                    logger.exception("model_repo_load_spec_failed", dir=candidate.name)

        self._last_scan = time.time()
