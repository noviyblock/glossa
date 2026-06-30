from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from config import DWPOSE_DET_CONF, DWPOSE_DET_NMS, DWPOSE_DET_PATH, DWPOSE_POSE_PATH

# Dataset keypoint layout (matches the offline extraction script used to build
# data/processed/ and train the ST-GCN classifier):
#   0-16   COCO-WholeBody body (17 pts)
#   17-22  COCO-WholeBody feet (6 pts)
#   23-32  duplicated key body joints, padding pose to 33 pts:
#          shoulders(5,6), hips(11,12), knees(13,14), ankles(15,16), shoulders(5,6) again
#   33-53  left hand  (COCO-WholeBody 91-111, 21 pts)
#   54-74  right hand (COCO-WholeBody 112-132, 21 pts)
# COCO-WholeBody face (23-90, 68 pts) is dropped entirely.
_N_POSE   = 33
_N_HAND   = 21
_TOTAL_KP = _N_POSE + _N_HAND + _N_HAND  # 75

_EXTRA_JOINT_SRC = [5, 6, 11, 12, 13, 14, 15, 16, 5, 6]  # → indices 23-32

_DET_INPUT_SIZE  = (640, 640)   # YOLOX (w, h)
_POSE_INPUT_SIZE = (288, 384)   # RTMPose/DWPose body+hand (w, h) — dw-ll_ucoco_384
_SIMCC_SPLIT_RATIO = 2.0


def _remap_coco133_to_75(kp133: np.ndarray) -> np.ndarray:
    """kp133: (133, 3) [x, y, score] in COCO-WholeBody order → (75, 3)."""
    body17 = kp133[0:17]
    feet6  = kp133[17:23]
    extra10 = kp133[_EXTRA_JOINT_SRC]
    pose33 = np.concatenate([body17, feet6, extra10], axis=0)
    left_hand  = kp133[91:112]
    right_hand = kp133[112:133]
    return np.concatenate([pose33, left_hand, right_hand], axis=0).astype(np.float32)


# ── YOLOX person detector ──────────────────────────────────────────────────── #

def _preprocess_det(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    in_w, in_h = input_size
    padded = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
    r = min(in_h / img.shape[0], in_w / img.shape[1])
    resized = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(blob[None]), r


def _yolox_grids(input_size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)):
    in_w, in_h = input_size
    grids, expanded_strides = [], []
    for stride in strides:
        gh, gw = in_h // stride, in_w // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        expanded_strides.append(np.full((gh * gw, 1), stride))
    return np.concatenate(grids, axis=0), np.concatenate(expanded_strides, axis=0)


def _decode_yolox(outputs: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    grids, strides = _yolox_grids(input_size)
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def _detect_person(
    session: ort.InferenceSession, img: np.ndarray, conf_thr: float, nms_thr: float
) -> np.ndarray | None:
    blob, ratio = _preprocess_det(img, _DET_INPUT_SIZE)
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: blob})[0][0]  # (N, 85): cx,cy,w,h,obj,cls...
    raw = _decode_yolox(raw, _DET_INPUT_SIZE)

    boxes_cxcywh = raw[:, :4]
    scores = raw[:, 4:5] * raw[:, 5:]
    person_scores = scores[:, 0]  # COCO class 0 = person

    mask = person_scores > conf_thr
    if not mask.any():
        return None

    cx, cy, w, h = boxes_cxcywh[mask].T
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) / ratio
    kept = _nms(boxes_xyxy, person_scores[mask], nms_thr)
    if not kept:
        return None

    best = kept[np.argmax(person_scores[mask][kept])]
    return boxes_xyxy[best]


# ── RTMPose / DWPose top-down pose estimator ───────────────────────────────── #

def _bbox_to_center_scale(bbox: np.ndarray, aspect_ratio: float, padding: float = 1.25):
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    w, h = x2 - x1, y2 - y1
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32) * padding
    return center, scale


def _crop_pose_input(
    img: np.ndarray, center: np.ndarray, scale: np.ndarray, input_size: tuple[int, int]
) -> np.ndarray:
    in_w, in_h = input_size
    src_w, src_h = scale
    dst = np.array(
        [[0, 0], [in_w, 0], [0, in_h]],
        dtype=np.float32,
    )
    src = np.array(
        [
            [center[0] - src_w / 2, center[1] - src_h / 2],
            [center[0] + src_w / 2, center[1] - src_h / 2],
            [center[0] - src_w / 2, center[1] + src_h / 2],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, matrix, input_size, flags=cv2.INTER_LINEAR)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
    """simcc_x: (K, Wx), simcc_y: (K, Wy) → (K, 3) [x, y, score] in crop-pixel coords."""
    x_locs = simcc_x.argmax(axis=1).astype(np.float32)
    y_locs = simcc_y.argmax(axis=1).astype(np.float32)
    x_vals = simcc_x.max(axis=1)
    y_vals = simcc_y.max(axis=1)
    scores = np.minimum(x_vals, y_vals)
    x_locs /= _SIMCC_SPLIT_RATIO
    y_locs /= _SIMCC_SPLIT_RATIO
    return np.stack([x_locs, y_locs, scores], axis=1)


def _estimate_pose(session: ort.InferenceSession, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Returns (133, 3) [x, y, score] in original-frame normalised coords."""
    aspect_ratio = _POSE_INPUT_SIZE[0] / _POSE_INPUT_SIZE[1]
    center, scale = _bbox_to_center_scale(bbox, aspect_ratio)
    crop = _crop_pose_input(img, center, scale, _POSE_INPUT_SIZE)

    blob = crop.transpose(2, 0, 1).astype(np.float32)[None]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    simcc_x, simcc_y = outputs[0][0], outputs[1][0]  # (133, Wx), (133, Wy)

    kp_crop = _decode_simcc(simcc_x, simcc_y)

    in_w, in_h = _POSE_INPUT_SIZE
    src_w, src_h = scale
    kp = kp_crop.copy()
    kp[:, 0] = kp_crop[:, 0] / in_w * src_w + (center[0] - src_w / 2)
    kp[:, 1] = kp_crop[:, 1] / in_h * src_h + (center[1] - src_h / 2)

    frame_h, frame_w = img.shape[:2]
    kp[:, 0] /= frame_w
    kp[:, 1] /= frame_h
    return kp


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class KeypointExtractor:
    """DWPose (YOLOX detector + RTMPose body+hand) keypoint extractor.

    Matches the offline pipeline used to build data/processed/ — same
    COCO-WholeBody → 75-point remap as mlops/training, so live inference
    stays consistent with the trained ST-GCN classifier.

    Person tracking: YOLOX detection runs every DET_INTERVAL frames; between
    detections the previous bounding box is reused.  This halves the per-frame
    cost (~170ms → ~90ms) at the price of slightly delayed bbox updates when
    the person moves significantly.
    """

    DET_INTERVAL = 5  # run YOLOX every N frames, track bbox in between

    def __init__(
        self,
        det_path: str = DWPOSE_DET_PATH,
        pose_path: str = DWPOSE_POSE_PATH,
        det_conf: float = DWPOSE_DET_CONF,
        det_nms: float = DWPOSE_DET_NMS,
    ) -> None:
        providers = ["CPUExecutionProvider"]
        self._det = ort.InferenceSession(det_path, providers=providers)
        self._pose = ort.InferenceSession(pose_path, providers=providers)
        self._det_conf = det_conf
        self._det_nms = det_nms
        # tracking state
        self._tracked_bbox: np.ndarray | None = None
        self._frames_since_det: int = 0

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return keypoints array of shape (75, 3) in [x, y, score] normalised coords.

        Returns all-zero keypoints if no person is detected in the frame.
        """
        self._frames_since_det += 1

        run_det = (self._tracked_bbox is None or
                   self._frames_since_det >= self.DET_INTERVAL)

        if run_det:
            new_bbox = _detect_person(self._det, bgr_frame, self._det_conf, self._det_nms)
            self._frames_since_det = 0
            if new_bbox is not None:
                # Accept new bbox if no prior track or IoU > 0.2 (person didn't teleport)
                if self._tracked_bbox is None or _bbox_iou(self._tracked_bbox, new_bbox) > 0.2:
                    self._tracked_bbox = new_bbox
                else:
                    # Large jump — reset to new detection (scene change / new person)
                    self._tracked_bbox = new_bbox
            else:
                # Person not found — keep last bbox for 1 more detection cycle then clear
                if self._frames_since_det == 0:
                    self._tracked_bbox = None

        if self._tracked_bbox is None:
            return np.zeros((_TOTAL_KP, 3), dtype=np.float32)

        kp133 = _estimate_pose(self._pose, bgr_frame, self._tracked_bbox)
        return _remap_coco133_to_75(kp133)

    def reset_tracking(self) -> None:
        """Force re-detection on the next frame (call when session resets)."""
        self._tracked_bbox = None
        self._frames_since_det = 0

    def close(self) -> None:
        pass
