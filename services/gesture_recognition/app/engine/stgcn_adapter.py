"""ST-GCN (Spatial Temporal Graph Convolutional Network) model adapter.

ST-GCN treats the human skeleton as a spatial graph and applies graph
convolutions across time.  Reference: Yan et al., 2018.

Input format expected by ST-GCN ONNX models:
  [B, C, T, V]
  B = batch size
  C = channels (3 for xyz, 4 with visibility)
  T = temporal frames
  V = vertices (joints): 33 pose + 21+21 hands = 75

Skeleton graph
--------------
We define two adjacency matrices:
  1. Spatial: anatomically connected joints
  2. Self-loop: every joint connected to itself

The adapter reshapes the [B, T, 258] feature vector (produced by MediaPipe)
into the [B, C, T, V] graph tensor that ST-GCN expects.

Feature vector layout (see domain/sliding_window.py FEATURE_DIM):
  [0:132]    pose landmarks × 4 (x, y, z, visibility)
  [132:195]  left hand landmarks × 3 (x, y, z)
  [195:258]  right hand landmarks × 3 (x, y, z)
"""

from __future__ import annotations

import numpy as np

from ..domain.entities import ModelSpec
from .base import BaseInferenceEngine, _softmax
from .onnx_engine import ONNXInferenceEngine
from .openvino_engine import OpenVINOInferenceEngine

# ── Skeleton topology ──────────────────────────────────────────────────────────
POSE_N = 33
HAND_N = 21
TOTAL_JOINTS = POSE_N + HAND_N * 2   # 75

# MediaPipe pose connections (joint index pairs)
_POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),           # face
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),                                   # mouth
    (11, 12),                                  # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # left arm
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # right arm
    (18, 20),
    (11, 23), (12, 24),                        # torso
    (23, 24), (23, 25), (24, 26),              # hips
    (25, 27), (27, 29), (29, 31), (31, 27),    # left leg
    (26, 28), (28, 30), (30, 32), (32, 28),    # right leg
]

# Hand connections (finger chains)
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),          # palm
]

# Wrist connections (pose joints → hand wrists)
_CROSS_EDGES = [(15, POSE_N), (16, POSE_N + HAND_N)]  # left wrist, right wrist


def build_adjacency_matrix(num_joints: int = TOTAL_JOINTS) -> np.ndarray:
    """Build normalized adjacency matrix for the full body skeleton."""
    adj = np.eye(num_joints, dtype=np.float32)   # self-loops

    def _add_edges(edges: list[tuple[int, int]], offset: int = 0) -> None:
        for i, j in edges:
            a, b = i + offset, j + offset
            adj[a, b] = 1.0
            adj[b, a] = 1.0

    _add_edges(_POSE_EDGES, 0)
    _add_edges(_HAND_EDGES, POSE_N)              # left hand
    _add_edges(_HAND_EDGES, POSE_N + HAND_N)     # right hand
    _add_edges(_CROSS_EDGES, 0)

    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    return adj


_ADJ = build_adjacency_matrix()


def feature_vector_to_stgcn(batch_seq: np.ndarray) -> np.ndarray:
    """Reshape [B, T, D=258] → [B, C=4, T, V=75] for ST-GCN.

    Channel layout:
      C=0: x
      C=1: y
      C=2: z
      C=3: visibility (0.0 for hands where not available)
    """
    B, T, D = batch_seq.shape

    # Split feature vector into components
    pose_flat  = batch_seq[:, :, :132].reshape(B, T, POSE_N, 4)  # x y z vis
    lhand_flat = batch_seq[:, :, 132:195].reshape(B, T, HAND_N, 3)
    rhand_flat = batch_seq[:, :, 195:258].reshape(B, T, HAND_N, 3)

    # Pad hands with a visibility channel (always 1.0)
    ones = np.ones((B, T, HAND_N, 1), dtype=np.float32)
    lhand_4 = np.concatenate([lhand_flat, ones], axis=-1)  # (B, T, 21, 4)
    rhand_4 = np.concatenate([rhand_flat, ones], axis=-1)

    # Concatenate all joints: [B, T, 75, 4]
    joints = np.concatenate([pose_flat, lhand_4, rhand_4], axis=2)

    # Transpose to [B, C, T, V]
    stgcn_input = joints.transpose(0, 3, 1, 2)  # (B, 4, T, 75)
    return stgcn_input.astype(np.float32)


class STGCNAdapter(BaseInferenceEngine):
    """ST-GCN adapter that wraps an underlying ONNX or OpenVINO engine.

    The adapter:
    1. Reshapes the [B, T, D] feature sequence to [B, C, T, V] graph input.
    2. Delegates the actual inference to the inner engine.
    3. Returns logits [B, num_classes].
    """

    def __init__(
        self,
        inner_engine: ONNXInferenceEngine | OpenVINOInferenceEngine,
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
    ) -> None:
        super().__init__(spec, labels, feature_dim)
        self._inner = inner_engine
        self._ready = inner_engine.is_ready
        self._adj = _ADJ

    async def _load(self) -> None:
        await self._inner._load()
        self._ready = True

    def _run_inference_sync(self, batch: np.ndarray) -> np.ndarray:
        """Reshape and delegate to inner engine."""
        stgcn_batch = feature_vector_to_stgcn(batch)  # [B, 4, T, 75]
        return self._inner._run_inference_sync(stgcn_batch)

    async def reload(self, new_spec: ModelSpec) -> None:
        await self._inner.reload(new_spec)
        self._spec = new_spec
