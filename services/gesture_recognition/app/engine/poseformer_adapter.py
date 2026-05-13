"""PoseFormer model adapter.

PoseFormer (Zheng et al., 2021) applies self-attention across time to
pose sequences.  The architecture is purely sequential — no graph structure.

Input format expected by PoseFormer ONNX models:
  [B, T, J * D_joint]
  B = batch
  T = time steps (e.g. 30)
  J = joints considered (pose only: 33, or pose+hands: 75)
  D_joint = features per joint (3 for xyz, 4 with visibility)

Unlike ST-GCN, PoseFormer does NOT need graph reshaping — the input is
already a 1D sequence of concatenated joint coordinates per frame.

We support two PoseFormer variants:
  "pose_only"   — uses only the 33 pose joints × 4 = 132 features
  "full_body"   — uses all 75 joints × 4 = 300 features

The adapter normalizes the feature vector to match the expected variant.
"""

from __future__ import annotations

import numpy as np

from ..domain.entities import ModelSpec
from .base import BaseInferenceEngine
from .onnx_engine import ONNXInferenceEngine
from .openvino_engine import OpenVINOInferenceEngine

_POSE_N = 33
_HAND_N = 21

# Feature slices within the standard 258-dim vector
_POSE_SLICE = slice(0, 132)      # 33 × 4
_LHAND_SLICE = slice(132, 195)   # 21 × 3
_RHAND_SLICE = slice(195, 258)   # 21 × 3


class PoseFormerVariant:
    POSE_ONLY = "pose_only"     # 132 features/frame
    FULL_BODY = "full_body"     # 258 features/frame (standard)
    HANDS_ONLY = "hands_only"   # 126 features/frame (gesture focus)


def select_features(
    batch_seq: np.ndarray, variant: str = PoseFormerVariant.FULL_BODY
) -> np.ndarray:
    """Slice feature channels based on PoseFormer variant.

    batch_seq: [B, T, 258]  →  [B, T, D_variant]
    """
    if variant == PoseFormerVariant.POSE_ONLY:
        return batch_seq[:, :, _POSE_SLICE]           # [B, T, 132]

    if variant == PoseFormerVariant.HANDS_ONLY:
        hands = batch_seq[:, :, 132:]                 # [B, T, 126]
        return hands

    return batch_seq                                  # [B, T, 258] full


def add_temporal_embedding(seq: np.ndarray) -> np.ndarray:
    """Add sinusoidal positional encoding to the temporal dimension.

    PoseFormer models trained without positional embeddings expect raw sequence.
    Models with learned PE already bake this into weights.
    This is applied only when spec.extra["temporal_pe"] == "sinusoidal".

    seq: [B, T, D] → [B, T, D]
    """
    B, T, D = seq.shape
    position = np.arange(T, dtype=np.float32)[:, None]   # (T, 1)
    div_term = np.exp(
        np.arange(0, D, 2, dtype=np.float32) * -(np.log(10000.0) / D)
    )

    pe = np.zeros((T, D), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: D // 2])

    return seq + pe[None, :, :]   # broadcast over batch


class PoseFormerAdapter(BaseInferenceEngine):
    """PoseFormer adapter — feature selection + optional PE injection.

    The inner engine handles hardware-specific inference.
    This adapter handles model-architecture-specific preprocessing.
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
        self._variant: str = spec.extra.get("variant", PoseFormerVariant.FULL_BODY)
        self._use_temporal_pe: bool = spec.extra.get("temporal_pe") == "sinusoidal"

    async def _load(self) -> None:
        await self._inner._load()
        self._ready = True

    def _run_inference_sync(self, batch: np.ndarray) -> np.ndarray:
        """Feature selection → optional PE → delegate to inner engine."""
        processed = select_features(batch, self._variant)

        if self._use_temporal_pe:
            processed = add_temporal_embedding(processed)

        return self._inner._run_inference_sync(processed)

    async def reload(self, new_spec: ModelSpec) -> None:
        await self._inner.reload(new_spec)
        self._spec = new_spec
        self._variant = new_spec.extra.get("variant", PoseFormerVariant.FULL_BODY)
        self._use_temporal_pe = new_spec.extra.get("temporal_pe") == "sinusoidal"
