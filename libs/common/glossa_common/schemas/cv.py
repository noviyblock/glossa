from uuid import UUID

from pydantic import Field

from .base import GlossaModel


class PoseLandmark(GlossaModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    z: float
    visibility: float = Field(default=0.0, ge=0.0, le=1.0)


class HandLandmarks(GlossaModel):
    landmarks: list[PoseLandmark] = Field(min_length=21, max_length=21)
    handedness: str = Field(pattern=r"^(Left|Right)$")
    confidence: float = Field(ge=0.0, le=1.0)


class FaceLandmarks(GlossaModel):
    landmarks: list[PoseLandmark]
    confidence: float = Field(ge=0.0, le=1.0)


class KeypointFrame(GlossaModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0)
    pose: list[PoseLandmark] | None = None           # 33 landmarks
    left_hand: HandLandmarks | None = None
    right_hand: HandLandmarks | None = None
    face: FaceLandmarks | None = None


class GestureRecognitionRequest(GlossaModel):
    session_id: UUID
    frames: list[KeypointFrame] = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0.0)


class GlossToken(GlossaModel):
    gloss: str
    start_frame: int
    end_frame: int
    confidence: float = Field(ge=0.0, le=1.0)


class GestureRecognitionResult(GlossaModel):
    session_id: UUID
    gloss_sequence: list[GlossToken]
    raw_text: str  # space-joined glosses
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float
    frames_processed: int
