from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.schemas.cv import FaceLandmarks, HandLandmarks, HolisticFrame, PoseLandmark

logger = get_logger(__name__)


def _mp_landmark_to_pose(lm: Any) -> PoseLandmark:
    return PoseLandmark(
        x=float(lm.x),
        y=float(lm.y),
        z=float(lm.z),
        visibility=float(getattr(lm, "visibility", 0.0)),
    )


class MediaPipeHolisticAdapter:
    """Wraps MediaPipe Holistic for async frame processing.

    MediaPipe is synchronous and not thread-safe per instance, so we run it
    in a dedicated thread pool with one instance per worker.
    """

    def __init__(self, model_complexity: int = 1, min_detection: float = 0.5, min_tracking: float = 0.5) -> None:
        self._complexity = model_complexity
        self._min_det = min_detection
        self._min_track = min_tracking
        self._holistic: Any | None = None

    def _init_holistic(self) -> None:
        import mediapipe as mp

        self._holistic = mp.solutions.holistic.Holistic(
            model_complexity=self._complexity,
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
            static_image_mode=False,
        )
        logger.info("mediapipe_holistic_initialized", complexity=self._complexity)

    async def ensure_initialized(self) -> None:
        if self._holistic is None:
            await asyncio.to_thread(self._init_holistic)

    def process_frame_sync(self, bgr_frame: np.ndarray, frame_index: int, timestamp_ms: float) -> HolisticFrame:
        if self._holistic is None:
            raise RuntimeError("MediaPipe not initialized — call ensure_initialized() first")

        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        pose = (
            [_mp_landmark_to_pose(lm) for lm in result.pose_landmarks.landmark]
            if result.pose_landmarks
            else None
        )
        left_hand = (
            HandLandmarks(
                landmarks=[_mp_landmark_to_pose(lm) for lm in result.left_hand_landmarks.landmark],
                handedness="Left",
                confidence=1.0,
            )
            if result.left_hand_landmarks
            else None
        )
        right_hand = (
            HandLandmarks(
                landmarks=[_mp_landmark_to_pose(lm) for lm in result.right_hand_landmarks.landmark],
                handedness="Right",
                confidence=1.0,
            )
            if result.right_hand_landmarks
            else None
        )
        face = (
            FaceLandmarks(
                landmarks=[_mp_landmark_to_pose(lm) for lm in result.face_landmarks.landmark],
                confidence=1.0,
            )
            if result.face_landmarks
            else None
        )

        return HolisticFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
        )

    async def process_frame(self, bgr_frame: np.ndarray, frame_index: int, timestamp_ms: float) -> HolisticFrame:
        await self.ensure_initialized()
        return await asyncio.to_thread(
            self.process_frame_sync, bgr_frame, frame_index, timestamp_ms
        )

    async def process_video_bytes(self, video_bytes: bytes) -> list[HolisticFrame]:
        import cv2
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        frames: list[HolisticFrame] = []
        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break
                ts_ms = (idx / fps) * 1000
                frame = await self.process_frame(bgr, frame_index=idx, timestamp_ms=ts_ms)
                frames.append(frame)
                idx += 1
            cap.release()
        finally:
            os.unlink(tmp_path)

        return frames

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
