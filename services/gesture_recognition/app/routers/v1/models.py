"""Model management endpoints — load, unload, hot-reload, set-active, list."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ...dependencies import get_registry
from ...domain.entities import ModelSpec
from ...registry.model_registry import ModelAlreadyLoadedError, ModelNotFoundError
from ...schemas.request import HotReloadRequest, LoadModelRequest, SetActiveRequest
from ...schemas.response import MessageResponse, ModelInfoResponse, ModelListResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def _spec_to_info(spec: ModelSpec, active_id: str | None) -> ModelInfoResponse:
    return ModelInfoResponse(
        id=spec.id,
        name=spec.name,
        architecture=spec.architecture,
        backend=spec.backend,
        quantization=spec.quantization,
        version=spec.version,
        status=spec.status,
        is_active=spec.id == active_id,
        loaded_at=spec.loaded_at,
        description=spec.description,
    )


@router.get("", response_model=ModelListResponse)
async def list_models(registry=Depends(get_registry)) -> ModelListResponse:
    specs = registry.list_all()
    return ModelListResponse(
        models=[_spec_to_info(s, registry.active_model_id) for s in specs],
        active_model_id=registry.active_model_id,
        loaded_count=registry.loaded_count,
    )


@router.post("/load", response_model=MessageResponse, status_code=201)
async def load_model(
    body: LoadModelRequest,
    registry=Depends(get_registry),
) -> MessageResponse:
    from ...domain.entities import ModelArchitecture, ModelBackend, QuantizationType

    spec = ModelSpec(
        id=body.model_id,
        name=body.name or body.model_id,
        architecture=ModelArchitecture(body.architecture),
        backend=ModelBackend(body.backend),
        quantization=QuantizationType(body.quantization),
        path=Path(body.path),
        labels_path=Path(body.labels_path),
        version=body.version,
        description=body.description,
    )

    try:
        await registry.load(spec)
    except ModelAlreadyLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Load failed: {exc}")

    if body.set_active:
        registry.set_active(body.model_id)

    return MessageResponse(message=f"Model '{body.model_id}' loaded successfully")


@router.delete("/{model_id}", response_model=MessageResponse)
async def unload_model(
    model_id: str,
    registry=Depends(get_registry),
) -> MessageResponse:
    try:
        await registry.unload(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return MessageResponse(message=f"Model '{model_id}' unloaded")


@router.post("/{model_id}/hot-reload", response_model=MessageResponse)
async def hot_reload(
    model_id: str,
    body: HotReloadRequest,
    registry=Depends(get_registry),
) -> MessageResponse:
    try:
        current_specs = {s.id: s for s in registry.list_all()}
        if model_id not in current_specs:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        old_spec = current_specs[model_id]
        import dataclasses
        new_spec = dataclasses.replace(
            old_spec,
            path=Path(body.new_path),
            version=body.new_version,
            labels_path=Path(body.labels_path) if body.labels_path else old_spec.labels_path,
        )
        await registry.hot_reload(model_id, new_spec)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hot-reload failed: {exc}")

    return MessageResponse(
        message=f"Model '{model_id}' hot-reloaded to version {body.new_version}"
    )


@router.post("/{model_id}/activate", response_model=MessageResponse)
async def set_active(
    model_id: str,
    registry=Depends(get_registry),
) -> MessageResponse:
    try:
        registry.set_active(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return MessageResponse(message=f"Active model set to '{model_id}'")


@router.get("/{model_id}", response_model=ModelInfoResponse)
async def get_model(
    model_id: str,
    registry=Depends(get_registry),
) -> ModelInfoResponse:
    try:
        specs = {s.id: s for s in registry.list_all()}
        if model_id not in specs:
            raise ModelNotFoundError(model_id)
        return _spec_to_info(specs[model_id], registry.active_model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
