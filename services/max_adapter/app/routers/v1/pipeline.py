from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import Field

from glossa_common.schemas.base import GlossaModel
from ..domain.adapter import MAXAdapterDomain, MAXPipelineRequest

router = APIRouter()


class PipelineRequest(GlossaModel):
    session_id: UUID = Field(default_factory=uuid4)
    model_name: str = Field(min_length=1, max_length=128)
    inputs: dict[str, object] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0.0, le=300.0)


@router.post(
    "/infer",
    summary="Run inference through MAX SDK pipeline",
    status_code=status.HTTP_200_OK,
)
async def run_inference(request: PipelineRequest, http_request: Request) -> dict:
    domain: MAXAdapterDomain = http_request.app.state.domain
    try:
        result = await domain.run(
            MAXPipelineRequest(
                session_id=request.session_id,
                model_name=request.model_name,
                inputs=request.inputs,
                timeout=request.timeout,
            )
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="MAX pipeline timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MAX inference error: {exc}") from exc

    return result.to_dict()
