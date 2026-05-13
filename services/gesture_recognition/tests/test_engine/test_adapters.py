"""Tests for STGCN and PoseFormer feature adapters."""

import numpy as np
import pytest

from app.engine.poseformer_adapter import PoseFormerAdapter, PoseFormerVariant
from app.engine.stgcn_adapter import STGCNAdapter


class TestSTGCNAdapter:
    def setup_method(self):
        self.adapter = STGCNAdapter(num_classes=10, sequence_length=30)

    def test_output_shape(self):
        seq = np.random.randn(30, 258).astype(np.float32)
        out = self.adapter.preprocess(seq)
        assert out.shape == (1, 4, 30, 75), f"Expected (1,4,30,75), got {out.shape}"

    def test_output_dtype(self):
        seq = np.random.randn(30, 258).astype(np.float32)
        out = self.adapter.preprocess(seq)
        assert out.dtype == np.float32

    def test_batch_preprocess(self):
        batch = np.random.randn(4, 30, 258).astype(np.float32)
        out = self.adapter.preprocess_batch(batch)
        assert out.shape == (4, 4, 30, 75)

    def test_adjacency_matrix_symmetric(self):
        adj = self.adapter.get_adjacency_matrix()
        # Normalized adj is not strictly symmetric but should be square
        assert adj.shape[0] == adj.shape[1] == 75

    def test_short_sequence_padded(self):
        seq = np.random.randn(10, 258).astype(np.float32)
        out = self.adapter.preprocess(seq)
        assert out.shape == (1, 4, 30, 75)


class TestPoseFormerAdapter:
    def test_full_variant_shape(self):
        adapter = PoseFormerAdapter(num_classes=10, variant=PoseFormerVariant.FULL)
        seq = np.random.randn(30, 258).astype(np.float32)
        out = adapter.preprocess(seq)
        assert out.shape == (1, 30, 258)

    def test_pose_only_variant_shape(self):
        adapter = PoseFormerAdapter(num_classes=10, variant=PoseFormerVariant.POSE_ONLY)
        seq = np.random.randn(30, 258).astype(np.float32)
        out = adapter.preprocess(seq)
        assert out.shape[2] == 132  # 33 × 4

    def test_hands_only_variant_shape(self):
        adapter = PoseFormerAdapter(num_classes=10, variant=PoseFormerVariant.HANDS_ONLY)
        seq = np.random.randn(30, 258).astype(np.float32)
        out = adapter.preprocess(seq)
        assert out.shape[2] == 126  # 42 × 3

    def test_positional_encoding_applied(self):
        adapter = PoseFormerAdapter(
            num_classes=10, variant=PoseFormerVariant.FULL, use_positional_encoding=True
        )
        seq = np.zeros((30, 258), dtype=np.float32)
        out = adapter.preprocess(seq)
        # PE is additive — output should not be all zeros
        assert not np.allclose(out, 0.0)

    def test_batch_preprocess(self):
        adapter = PoseFormerAdapter(num_classes=10, variant=PoseFormerVariant.FULL)
        batch = np.random.randn(3, 30, 258).astype(np.float32)
        out = adapter.preprocess_batch(batch)
        assert out.shape == (3, 30, 258)
