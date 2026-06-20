from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

# 33 pose + 21 left hand + 21 right hand = 75 keypoints
_N_POSE       = 33
_N_HAND       = 21
_TOTAL_KP     = _N_POSE + _N_HAND + _N_HAND  # 75


class KeypointExtractor:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return keypoints array of shape (75, 3) in [x, y, z] normalised coords.

        Indices:
            0-32   — pose landmarks (33 pts)
            33-53  — left hand landmarks (21 pts)
            54-74  — right hand landmarks (21 pts)

        Missing landmarks are filled with zeros.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        kp = np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                kp[i] = [lm.x, lm.y, lm.z]

        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks.landmark):
                kp[_N_POSE + i] = [lm.x, lm.y, lm.z]

        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks.landmark):
                kp[_N_POSE + _N_HAND + i] = [lm.x, lm.y, lm.z]

        return kp

    def close(self) -> None:
        self._holistic.close()
