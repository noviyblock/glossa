"""Gesture recognition endpoint."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...dependencies import get_recognition_service
from ...schemas.request import RecognizeRequest
from ...schemas.response import GlossCandidateResponse, RecognizeResponse
from ...service.recognition_service import GestureRecognitionService

router = APIRouter(prefix="/api/v1", tags=["recognition"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    body: RecognizeRequest,
    svc: GestureRecognitionService = Depends(get_recognition_service),
) -> RecognizeResponse:
    """Run gesture recognition on a pre-extracted feature sequence.

    The feature_sequence must be a [T, D] matrix where:
    - T = number of frames (≥ 1)
    - D = feature dimension (typically 258: 33×4 pose + 21×3 left hand + 21×3 right hand)
    """
    feature_matrix = np.array(body.feature_sequence, dtype=np.float32)

    try:
        output = await svc.recognize(
            feature_sequence=feature_matrix,
            session_id=body.session_id,
            top_k=body.top_k,
            confidence_threshold=body.confidence_threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Flatten candidates from all windows
    all_candidates: list[GlossCandidateResponse] = []
    for result in output.candidates:
        for cand in result.candidates:
            all_candidates.append(GlossCandidateResponse(
                gloss=cand.gloss,
                confidence=round(cand.confidence, 4),
                gloss_id=cand.gloss_id,
                start_frame=cand.start_frame,
                end_frame=cand.end_frame,
            ))

    return RecognizeResponse(
        session_id=output.session_id,
        glosses=output.glosses,
        confidences=[round(c, 4) for c in output.confidences],
        candidates=all_candidates,
        is_fallback=output.is_fallback,
        inference_ms=round(output.inference_ms, 3),
        queue_wait_ms=round(output.queue_wait_ms, 3),
        model_id=output.model_id,
    )


@router.post("/recognize/reset")
async def reset_session(
    session_id: str = "",
    svc: GestureRecognitionService = Depends(get_recognition_service),
) -> dict:
    """Reset sliding window state for a session."""
    svc.reset_session(session_id)
    return {"message": "session reset", "session_id": session_id}
