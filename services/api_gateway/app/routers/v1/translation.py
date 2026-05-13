import time
from uuid import UUID

import orjson
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from glossa_common.schemas.base import ErrorDetail, ErrorResponse
from glossa_common.schemas.translation import TranslationMode, TranslationRequest, TranslationResult

from ...dependencies import HttpClientDep, PublisherDep, SettingsDep
from ...infrastructure.orchestrator import TranslationOrchestrator

router = APIRouter()


def _get_orchestrator(client: HttpClientDep, settings: SettingsDep) -> TranslationOrchestrator:
    return TranslationOrchestrator(client, settings)


@router.post(
    "/translate",
    response_model=TranslationResult,
    status_code=status.HTTP_200_OK,
    summary="Synchronous translation endpoint (max ~10s audio)",
)
async def translate(
    request: TranslationRequest,
    client: HttpClientDep,
    settings: SettingsDep,
    publisher: PublisherDep,
) -> TranslationResult:
    orchestrator = TranslationOrchestrator(client, settings)
    try:
        return await orchestrator.run(request)
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Translation pipeline timed out",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream service error: {exc}",
        ) from exc


@router.get(
    "/sessions/{session_id}",
    summary="Get translation session result by ID",
)
async def get_session(session_id: UUID, client: HttpClientDep, settings: SettingsDep) -> JSONResponse:
    orchestrator = TranslationOrchestrator(client, settings)
    result = await orchestrator.get_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content=orjson.loads(result.model_dump_json()))
