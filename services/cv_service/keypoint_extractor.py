from __future__ import annotations

import logging

import numpy as np

from config import RTMLIB_DEVICE, RTMLIB_MODE

# Dataset keypoint layout (matches offline extraction + ST-GCN training):
#   0-16   COCO-WholeBody body  (17 pts)
#   17-22  COCO-WholeBody feet  (6 pts)
#   23-32  duplicated key body joints to pad pose to 33 pts:
#          shoulders(5,6), hips(11,12), knees(13,14), ankles(15,16), shoulders(5,6)
#   33-53  left hand  (COCO-WholeBody 91-111, 21 pts)
#   54-74  right hand (COCO-WholeBody 112-132, 21 pts)
_N_POSE   = 33
_N_HAND   = 21
_TOTAL_KP = _N_POSE + _N_HAND + _N_HAND  # 75

_EXTRA_JOINT_SRC = [5, 6, 11, 12, 13, 14, 15, 16, 5, 6]  # → indices 23-32

logger = logging.getLogger(__name__)


def _remap_coco133_to_75(kp133: np.ndarray) -> np.ndarray:
    """kp133: (133, 3) [x, y, score] in COCO-WholeBody order → (75, 3)."""
    body17     = kp133[0:17]
    feet6      = kp133[17:23]
    extra10    = kp133[_EXTRA_JOINT_SRC]
    pose33     = np.concatenate([body17, feet6, extra10], axis=0)
    left_hand  = kp133[91:112]
    right_hand = kp133[112:133]
    return np.concatenate([pose33, left_hand, right_hand], axis=0).astype(np.float32)


class KeypointExtractor:
    """rtmlib Wholebody keypoint extractor.

    Uses RTMDet-nano (~10 ms) + RTMW pose model (~80 ms) = ~90 ms/frame,
    replacing the previous YOLOX-L (~100 ms) + DWPose (~70 ms) = ~170 ms pipeline.
    ONNX models are downloaded from HuggingFace on first use and cached in
    ~/.rtmlib/ (mounted as a persistent Docker volume).
    """

    def __init__(self, mode: str = RTMLIB_MODE, device: str = RTMLIB_DEVICE) -> None:
        from rtmlib import Wholebody  # deferred import — download happens here

        logger.info("Loading rtmlib Wholebody mode=%s device=%s", mode, device)
        self._model = Wholebody(
            mode=mode,
            backend="onnxruntime",
            device=device,
            to_openpose=False,  # keep COCO-WholeBody 133-kp format
        )
        logger.info("rtmlib Wholebody ready")

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return (75, 3) [x, y, score] keypoints in normalised [0, 1] coords."""
        h, w = bgr_frame.shape[:2]

        try:
            keypoints, scores = self._model(bgr_frame)
        except Exception as exc:
            logger.warning("rtmlib inference error: %s", exc)
            return np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        if keypoints is None or len(keypoints) == 0:
            return np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        # Pick the most-confident person when multiple are detected
        best = int(np.argmax([s.mean() for s in scores])) if len(keypoints) > 1 else 0
        kp = np.asarray(keypoints[best], dtype=np.float32)  # (K, 2) pixel coords
        sc = np.asarray(scores[best],    dtype=np.float32)   # (K,)

        K = kp.shape[0]
        if K < 133:
            kp = np.pad(kp, ((0, 133 - K), (0, 0)))
            sc = np.pad(sc, (0, 133 - K))
        elif K > 133:
            kp, sc = kp[:133], sc[:133]

        # Normalise pixel → [0, 1]
        kp[:, 0] = np.clip(kp[:, 0] / w, 0.0, 1.0)
        kp[:, 1] = np.clip(kp[:, 1] / h, 0.0, 1.0)

        kp133 = np.stack([kp[:, 0], kp[:, 1], sc], axis=1)  # (133, 3)
        return _remap_coco133_to_75(kp133)

    def reset_tracking(self) -> None:
        """No-op: RTMDet-nano is fast enough to run every frame."""

    def close(self) -> None:
        pass
