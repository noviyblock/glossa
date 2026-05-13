"""Tests for ModelRegistry — load, unload, hot-reload, set-active."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.domain.entities import ModelStatus
from app.registry.model_registry import (
    ModelAlreadyLoadedError,
    ModelNotFoundError,
    ModelRegistry,
)

from ..conftest import MockInferenceEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _registry_with_model(fake_spec, labels):
    """Return a registry with fake_spec already loaded using a mock engine."""
    registry = ModelRegistry(labels=labels, warmup_iterations=1)

    engine = MockInferenceEngine(labels=labels)
    with patch("app.registry.model_registry.create_engine", new=AsyncMock(return_value=engine)):
        await registry.load(fake_spec)

    return registry, engine


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_sets_active_for_first_model(fake_spec, labels):
    registry, _ = await _registry_with_model(fake_spec, labels)
    assert registry.active_model_id == fake_spec.id
    assert fake_spec.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_load_duplicate_raises(fake_spec, labels):
    registry, _ = await _registry_with_model(fake_spec, labels)
    with pytest.raises(ModelAlreadyLoadedError):
        await registry.load(fake_spec)


@pytest.mark.asyncio
async def test_get_active_returns_engine(fake_spec, labels):
    registry, engine = await _registry_with_model(fake_spec, labels)
    active = registry.get_active()
    assert active is engine


@pytest.mark.asyncio
async def test_get_active_raises_when_empty(labels):
    registry = ModelRegistry(labels=labels)
    with pytest.raises(RuntimeError, match="No active model"):
        registry.get_active()


@pytest.mark.asyncio
async def test_unload_removes_engine(fake_spec, labels):
    registry, _ = await _registry_with_model(fake_spec, labels)

    # Load a second model so we can unload the first
    import dataclasses
    spec2 = dataclasses.replace(fake_spec, id="model-2")
    engine2 = MockInferenceEngine(labels=labels, model_id="model-2")
    with patch("app.registry.model_registry.create_engine", new=AsyncMock(return_value=engine2)):
        await registry.load(spec2)

    registry.set_active("model-2")
    await registry.unload(fake_spec.id)
    assert not registry.has_model(fake_spec.id)


@pytest.mark.asyncio
async def test_unload_active_raises(fake_spec, labels):
    registry, _ = await _registry_with_model(fake_spec, labels)
    with pytest.raises(RuntimeError, match="Cannot unload the active model"):
        await registry.unload(fake_spec.id)


@pytest.mark.asyncio
async def test_set_active_unknown_raises(fake_spec, labels):
    registry, _ = await _registry_with_model(fake_spec, labels)
    with pytest.raises(ModelNotFoundError):
        registry.set_active("nonexistent-model")


@pytest.mark.asyncio
async def test_hot_reload_swaps_engine(fake_spec, labels):
    registry, old_engine = await _registry_with_model(fake_spec, labels)

    import dataclasses
    new_spec = dataclasses.replace(fake_spec, version="2.0.0")
    new_engine = MockInferenceEngine(labels=labels, model_id=fake_spec.id, top_confidence=0.99)

    with patch("app.registry.model_registry.create_engine", new=AsyncMock(return_value=new_engine)):
        await registry.hot_reload(fake_spec.id, new_spec)

    active = registry.get_active()
    assert active is new_engine


@pytest.mark.asyncio
async def test_load_many_concurrent(labels, fake_spec, tmp_path):
    import dataclasses

    specs = [
        dataclasses.replace(fake_spec, id=f"model-{i}")
        for i in range(3)
    ]

    engines = [MockInferenceEngine(labels=labels, model_id=s.id) for s in specs]
    call_iter = iter(engines)

    async def _create(spec, *args, **kwargs):
        return next(call_iter)

    registry = ModelRegistry(labels=labels, warmup_iterations=1)
    with patch("app.registry.model_registry.create_engine", side_effect=_create):
        await registry.load_many(specs)

    assert registry.loaded_count == 3
