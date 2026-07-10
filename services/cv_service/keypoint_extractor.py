from __future__ import annotations

import logging

import numpy as np

from config import (
    DET_INTERVAL,
    HAND_LOW_CONF_ZERO_THRESHOLD,
    LOW_CONF_ZERO_THRESHOLD,
    RTMLIB_DEVICE,
    RTMLIB_MODE,
    TRACK_CROP_PADDING,
)

# Dataset keypoint layout (matches offline extraction + ST-GCN training).
#
# The offline extraction script (colab_glossa_00_preprocess_dwpose_200.ipynb)
# used controlnet_aux's DwposeDetector, which returns body keypoints as
# OpenPose-18 JSON (pose_keypoints_2d) and remaps them into a "COCO-17" array
# via op_to_coco = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]. That produces
# a DIFFERENT ordering than the literal COCO-WholeBody body layout that
# rtmlib/DWPose ONNX models output directly:
#   training body17 = [nose, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
#                       Rhip, Rkne, Rank, Lhip, Lkne, Lank, Reye, Leye, Rear, Lear]
#   true COCO-17    = [nose, Leye, Reye, Lear, Rear, Lsho, Rsho,
#                       Lelb, Relb, Lwri, Rwri, Lhip, Rhip, Lkne, Rkne, Lank, Rank]
# _COCO17_TO_TRAINING_ORDER permutes true-COCO17 into the order ST-GCN was
# trained on, so live inference matches the offline features bit-for-bit.
#
#   0-16   body (17 pts, training order — see above)
#   17-22  COCO-WholeBody feet  (6 pts)
#   23-32  duplicated key body joints to pad pose to 33 pts (indices into the
#          training-order body17: originally intended as shoulders/hips/knees/
#          ankles, but since body17 is in training order these actually pick
#          out elbows/wrists/knees/ankles/eyes/ears — kept as-is to match
#          what the offline script produced, since consistency with training
#          data matters more than the (stale) anatomical naming)
#   33-53  left hand  (COCO-WholeBody 91-111, 21 pts)
#   54-74  right hand (COCO-WholeBody 112-132, 21 pts)
_N_POSE   = 33
_N_HAND   = 21
_TOTAL_KP = _N_POSE + _N_HAND + _N_HAND  # 75

_COCO17_TO_TRAINING_ORDER = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_EXTRA_JOINT_SRC = [5, 6, 11, 12, 13, 14, 15, 16, 5, 6]  # → indices 23-32

logger = logging.getLogger(__name__)


def _zero_low_confidence_joints(kp: np.ndarray, threshold: float = LOW_CONF_ZERO_THRESHOLD) -> np.ndarray:
    """Hard-zero [x, y, score] for any joint below `threshold`.

    The offline training pipeline (DWPose via controlnet_aux) used a separate
    per-hand detector that either finds a hand or doesn't — an undetected
    hand is a genuine (0, 0, 0) in the training data. rtmlib's RTMW is a
    single holistic top-down model that ALWAYS emits a coordinate for every
    joint (even occluded/out-of-frame ones), just with a lower confidence —
    it never produces a true "absent" reading. Left unzeroed, this creates
    phantom limbs (e.g. a "detected" second hand that isn't actually in
    frame) that the classifier never saw an equivalent of at train time.
    Explicitly zeroing reproduces the same hard-absent semantics.
    """
    kp = kp.copy()
    kp[kp[:, 2] < threshold] = 0.0
    return kp


def _remap_coco133_to_75(kp133: np.ndarray) -> np.ndarray:
    """kp133: (133, 3) [x, y, score] in COCO-WholeBody order → (75, 3)."""
    body17     = _zero_low_confidence_joints(kp133[0:17][_COCO17_TO_TRAINING_ORDER])
    feet6      = _zero_low_confidence_joints(kp133[17:23])
    extra10    = body17[_EXTRA_JOINT_SRC]  # derived from already-zeroed body17
    pose33     = np.concatenate([body17, feet6, extra10], axis=0)
    # Hands use their own, lower threshold — see HAND_LOW_CONF_ZERO_THRESHOLD.
    left_hand  = _zero_low_confidence_joints(kp133[91:112],  threshold=HAND_LOW_CONF_ZERO_THRESHOLD)
    right_hand = _zero_low_confidence_joints(kp133[112:133], threshold=HAND_LOW_CONF_ZERO_THRESHOLD)
    return np.concatenate([pose33, left_hand, right_hand], axis=0).astype(np.float32)


class TrackState:
    """Per-session bbox-tracking state for KeypointExtractor.extract().

    Deliberately NOT stored on KeypointExtractor itself: the extractor wraps
    one shared rtmlib model instance used concurrently by every session
    (`main.py` calls `extract()` via `asyncio.to_thread` for whichever
    session_id's request happens to be in flight). If tracking state lived
    on `self`, concurrent sessions would stomp on each other's bbox/counter
    between `await` points. Callers keep one TrackState per session_id (see
    `_session_tracks` in main.py) and pass it in explicitly.
    """

    __slots__ = ("bbox", "frames_since_det")

    def __init__(self) -> None:
        self.bbox: np.ndarray | None = None  # [x1,y1,x2,y2], full-frame pixel coords
        self.frames_since_det = 0

    def reset(self) -> None:
        self.bbox = None
        self.frames_since_det = 0


class KeypointExtractor:
    """rtmlib Wholebody keypoint extractor.

    Uses RTMDet-nano (~10 ms) + RTMW pose model (~80 ms) = ~90 ms/frame,
    replacing the previous YOLOX-L (~100 ms) + DWPose (~70 ms) = ~170 ms pipeline.
    ONNX models are downloaded from HuggingFace on first use and cached in
    ~/.rtmlib/ (mounted as a persistent Docker volume).

    Bbox tracking: rtmlib's Wholebody runs its own RTMDet-nano detector on
    whatever image it's given, every call, with no tracking of its own. We
    feed it the FULL frame every DET_INTERVAL calls (to (re)acquire the
    person), and a padded crop around the last known keypoint bounding box
    on the frames in between — both detection and pose estimation cost scale
    with input resolution, so a person-sized crop is meaningfully cheaper
    than the full camera frame. This works entirely from the outside (we
    just control what image we hand to `self._model(...)`), no dependency on
    rtmlib internals beyond the documented `Wholebody(image) -> (keypoints,
    scores)` contract, with keypoints in the pixel space of whatever image
    was passed in. Tracking state itself is per-session — see `TrackState`.
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

    def extract(self, bgr_frame: np.ndarray, track: TrackState | None = None) -> np.ndarray:
        """Return (75, 3) [x, y, score] keypoints in normalised [0, 1] coords.

        `track`: per-session TrackState (see class docstring). If None,
        every call runs on the full frame (no tracking) — safe stateless
        fallback, just without the bbox-crop speedup.
        """
        h, w = bgr_frame.shape[:2]

        if track is None:
            crop, offset, scale = bgr_frame, (0.0, 0.0), 1.0
        else:
            track.frames_since_det += 1
            use_full_frame = track.bbox is None or track.frames_since_det >= DET_INTERVAL
            if use_full_frame:
                crop, offset, scale = bgr_frame, (0.0, 0.0), 1.0
                track.frames_since_det = 0
            else:
                crop, offset, scale = self._crop_around(bgr_frame, track.bbox)

        try:
            keypoints, scores = self._model(crop)
        except Exception as exc:
            logger.warning("rtmlib inference error: %s", exc)
            if track is not None:
                track.reset()
            return np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        if keypoints is None or len(keypoints) == 0:
            # Nothing found in the (possibly cropped) region — force a
            # full-frame re-detect next call rather than tracking a stale bbox.
            if track is not None:
                track.reset()
            return np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        # Pick the most-confident person when multiple are detected
        best = int(np.argmax([s.mean() for s in scores])) if len(keypoints) > 1 else 0
        kp = np.asarray(keypoints[best], dtype=np.float32)  # (K, 2) pixel coords, in `crop` space
        sc = np.asarray(scores[best],    dtype=np.float32)   # (K,)

        K = kp.shape[0]
        if K < 133:
            kp = np.pad(kp, ((0, 133 - K), (0, 0)))
            sc = np.pad(sc, (0, 133 - K))
        elif K > 133:
            kp, sc = kp[:133], sc[:133]

        # crop-space → full-frame pixel coords
        kp[:, 0] = kp[:, 0] / scale + offset[0]
        kp[:, 1] = kp[:, 1] / scale + offset[1]

        if track is not None:
            self._update_tracked_bbox(track, kp, sc, w, h)

        # Normalise pixel → [0, 1]
        kp[:, 0] = np.clip(kp[:, 0] / w, 0.0, 1.0)
        kp[:, 1] = np.clip(kp[:, 1] / h, 0.0, 1.0)

        kp133 = np.stack([kp[:, 0], kp[:, 1], sc], axis=1)  # (133, 3)
        return _remap_coco133_to_75(kp133)

    @staticmethod
    def _crop_around(
        frame: np.ndarray, bbox: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        """Padded crop around `bbox` ([x1,y1,x2,y2], full-frame pixel coords).

        Returns (crop, (offset_x, offset_y), scale=1.0) — scale is always 1.0
        since we crop rather than resize (resizing would need rtmlib's own
        preprocessing to know the new effective resolution, which we don't
        control; a plain crop keeps the pixel-coordinate math trivial).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        half_w, half_h = bw * TRACK_CROP_PADDING / 2, bh * TRACK_CROP_PADDING / 2

        cx1 = int(np.clip(cx - half_w, 0, w - 1))
        cy1 = int(np.clip(cy - half_h, 0, h - 1))
        cx2 = int(np.clip(cx + half_w, cx1 + 1, w))
        cy2 = int(np.clip(cy + half_h, cy1 + 1, h))

        return frame[cy1:cy2, cx1:cx2], (float(cx1), float(cy1)), 1.0

    @staticmethod
    def _update_tracked_bbox(track: TrackState, kp: np.ndarray, sc: np.ndarray, w: int, h: int) -> None:
        """Grow `track.bbox` to include this frame's confidently-detected
        keypoints (full-frame pixel coords), for next call's crop.

        Deliberately a UNION with the previous bbox, not a replacement.
        Replacing it every call created a feedback loop: a hand swung out
        mid-gesture whose confidence dips below threshold (fast motion —
        rtmlib is a single-shot per-frame model, no temporal smoothing of
        its own) would drop out of THIS frame's confident-point set, so the
        recomputed bbox would shrink to exclude it — and the NEXT (cropped,
        non-full-frame) call would then not even show rtmlib that region of
        the image, guaranteeing it stays low/zero for up to DET_INTERVAL-1
        more frames. Only growing the box between full-frame redetects (which
        happen unconditionally every DET_INTERVAL calls and reset the box
        from scratch) fixes this while still bounding how large the crop can
        get.
        """
        valid = sc >= LOW_CONF_ZERO_THRESHOLD
        if not valid.any():
            return  # keep the existing bbox rather than dropping tracking entirely
        xs, ys = kp[valid, 0], kp[valid, 1]
        new_bbox = np.array(
            [max(xs.min(), 0), max(ys.min(), 0), min(xs.max(), w), min(ys.max(), h)],
            dtype=np.float32,
        )
        if track.bbox is None:
            track.bbox = new_bbox
        else:
            track.bbox = np.array(
                [min(track.bbox[0], new_bbox[0]), min(track.bbox[1], new_bbox[1]),
                 max(track.bbox[2], new_bbox[2]), max(track.bbox[3], new_bbox[3])],
                dtype=np.float32,
            )

    def close(self) -> None:
        pass
