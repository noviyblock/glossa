from fastapi import APIRouter, HTTPException, Request, status

from glossa_common.schemas.cv import GestureRecognitionRequest, GestureRecognitionResult

router = APIRouter()


@router.post(
    "/recognize",
    response_model=GestureRecognitionResult,
    status_code=status.HTTP_200_OK,
    summary="Recognize RSL gesture sequence from holistic landmark frames",
)
async def recognize_gestures(
    request: GestureRecognitionRequest,
    http_request: Request,
) -> GestureRecognitionResult:
    pipeline = http_request.app.state.pipeline
    try:
        return await pipeline.process(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV pipeline error: {exc}",
        ) from exc


@router.post(
    "/extract-landmarks",
    summary="Extract MediaPipe Holistic landmarks from raw video bytes",
)
async def extract_landmarks(
    http_request: Request,
) -> dict:
    mediapipe = http_request.app.state.mediapipe
    body = await http_request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No video data provided")

    frames = await mediapipe.process_video_bytes(body)
    return {
        "frames_extracted": len(frames),
        "frames": [f.model_dump() for f in frames],
    }
