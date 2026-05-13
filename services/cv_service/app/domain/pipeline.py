import time
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.schemas.cv import (
    FaceLandmarks,
    GestureRecognitionRequest,
    GestureRecognitionResult,
    GlossToken,
    HandLandmarks,
    HolisticFrame,
    PoseLandmark,
)

logger = get_logger(__name__)

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = 468
FEATURE_DIM = NUM_POSE * 4 + NUM_HAND * 3 * 2 + NUM_FACE * 3  # pose(xyzv) + hands(xyz*2) + face(xyz)


def _landmarks_to_array(landmarks: list[PoseLandmark], has_visibility: bool = False) -> np.ndarray:
    rows = []
    for lm in landmarks:
        row = [lm.x, lm.y, lm.z]
        if has_visibility:
            row.append(lm.visibility)
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def frame_to_feature_vector(frame: HolisticFrame) -> np.ndarray:
    pose_arr = (
        _landmarks_to_array(frame.pose, has_visibility=True).flatten()
        if frame.pose
        else np.zeros(NUM_POSE * 4, dtype=np.float32)
    )
    lh_arr = (
        _landmarks_to_array(frame.left_hand.landmarks).flatten()
        if frame.left_hand
        else np.zeros(NUM_HAND * 3, dtype=np.float32)
    )
    rh_arr = (
        _landmarks_to_array(frame.right_hand.landmarks).flatten()
        if frame.right_hand
        else np.zeros(NUM_HAND * 3, dtype=np.float32)
    )
    face_arr = (
        _landmarks_to_array(frame.face.landmarks).flatten()
        if frame.face
        else np.zeros(NUM_FACE * 3, dtype=np.float32)
    )
    return np.concatenate([pose_arr, lh_arr, rh_arr, face_arr])


class GestureClassifier(ABC):
    @abstractmethod
    async def predict(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        """Return (gloss_labels, confidences) for a [T, feature_dim] array."""

    @abstractmethod
    async def warmup(self) -> None:
        """Run a dummy inference to warm up the runtime."""


class CVPipeline:
    def __init__(self, classifier: GestureClassifier, sequence_length: int = 30, stride: int = 15) -> None:
        self._classifier = classifier
        self._seq_len = sequence_length
        self._stride = stride

    async def warmup(self) -> None:
        await self._classifier.warmup()

    async def process(self, request: GestureRecognitionRequest) -> GestureRecognitionResult:
        t0 = time.perf_counter()
        frames = request.frames

        features = np.stack([frame_to_feature_vector(f) for f in frames], axis=0)  # [T, D]

        gloss_tokens: list[GlossToken] = []
        i = 0
        while i + self._seq_len <= len(features):
            window = features[i : i + self._seq_len]
            labels, confs = await self._classifier.predict(window)
            for label, conf in zip(labels, confs):
                if label != "<blank>" and conf > 0.4:
                    gloss_tokens.append(
                        GlossToken(
                            gloss=label,
                            start_frame=frames[i].frame_index,
                            end_frame=frames[min(i + self._seq_len - 1, len(frames) - 1)].frame_index,
                            confidence=conf,
                        )
                    )
            i += self._stride

        # Deduplicate consecutive identical glosses
        deduped: list[GlossToken] = []
        for token in gloss_tokens:
            if not deduped or deduped[-1].gloss != token.gloss:
                deduped.append(token)

        avg_conf = float(np.mean([t.confidence for t in deduped])) if deduped else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "cv_processed",
            frames=len(frames),
            glosses=len(deduped),
            latency_ms=round(elapsed_ms, 2),
        )

        return GestureRecognitionResult(
            session_id=request.session_id,
            gloss_sequence=deduped,
            raw_text=" ".join(t.gloss for t in deduped),
            confidence=round(avg_conf, 4),
            processing_time_ms=round(elapsed_ms, 2),
            frames_processed=len(frames),
        )
