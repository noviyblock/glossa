"""Tests for audio buffer implementations."""

import numpy as np
import pytest

from app.domain.audio_buffer import ChunkedAudioBuffer, RingAudioBuffer


class TestRingAudioBuffer:
    def test_push_and_read(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000)
        audio = np.ones(16000, dtype=np.float32) * 0.5
        buf.push(audio)
        assert buf.n_samples == 16000

    def test_flush_returns_all(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000, overlap_s=0.0)
        audio = np.arange(1000, dtype=np.float32)
        buf.push(audio)
        out = buf.flush()
        assert len(out) == 1000
        np.testing.assert_array_equal(out, audio)

    def test_overflow_keeps_last_capacity(self):
        buf = RingAudioBuffer(capacity_s=1.0, sample_rate=16000, overlap_s=0.0)
        big = np.ones(32000, dtype=np.float32)  # 2 × capacity
        buf.push(big)
        assert buf.n_samples == 16000

    def test_ready_to_emit(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000, emit_threshold_s=1.0)
        buf.push(np.zeros(8000, dtype=np.float32))   # 0.5 s < threshold
        assert not buf.ready_to_emit
        buf.push(np.zeros(8001, dtype=np.float32))   # now > 1.0 s
        assert buf.ready_to_emit

    def test_peek_no_consume(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000, overlap_s=0.0)
        audio = np.linspace(0, 1, 1600, dtype=np.float32)
        buf.push(audio)
        peeked = buf.peek(1600)
        assert len(peeked) == 1600
        assert buf.n_samples == 1600  # not consumed

    def test_reset_clears_buffer(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000)
        buf.push(np.ones(16000, dtype=np.float32))
        buf.reset()
        assert buf.n_samples == 0

    def test_flush_preserves_overlap(self):
        buf = RingAudioBuffer(capacity_s=5.0, sample_rate=16000, overlap_s=0.5)
        buf.push(np.ones(32000, dtype=np.float32))
        buf.flush()
        assert buf.n_samples == 8000  # 0.5 s overlap preserved

    def test_wrap_around_integrity(self):
        buf = RingAudioBuffer(capacity_s=1.0, sample_rate=1000, overlap_s=0.0)
        for i in range(3):
            chunk = np.full(400, float(i), dtype=np.float32)
            buf.push(chunk)
        out = buf.flush()
        assert len(out) == 1000  # full capacity


class TestChunkedAudioBuffer:
    def test_no_emit_below_max(self):
        buf = ChunkedAudioBuffer(sample_rate=16000, max_chunk_s=30.0, min_chunk_s=0.5)
        chunks = buf.push(np.zeros(16000, dtype=np.float32))  # 1 s
        assert chunks == []

    def test_emits_at_max_chunk(self):
        buf = ChunkedAudioBuffer(sample_rate=16000, max_chunk_s=1.0, min_chunk_s=0.1, overlap_s=0.0)
        audio = np.zeros(32000, dtype=np.float32)   # 2 s → should emit 2 chunks
        chunks = buf.push(audio)
        assert len(chunks) == 2
        assert len(chunks[0]) == 16000

    def test_flush_returns_remainder(self):
        buf = ChunkedAudioBuffer(sample_rate=16000, max_chunk_s=10.0, min_chunk_s=0.1, overlap_s=0.0)
        buf.push(np.zeros(8000, dtype=np.float32))   # 0.5 s
        remainder = buf.flush()
        assert remainder is not None
        assert len(remainder) == 8000

    def test_flush_none_if_too_short(self):
        buf = ChunkedAudioBuffer(sample_rate=16000, max_chunk_s=10.0, min_chunk_s=1.0)
        buf.push(np.zeros(100, dtype=np.float32))
        remainder = buf.flush()
        assert remainder is None

    def test_overlap_keeps_context(self):
        buf = ChunkedAudioBuffer(sample_rate=1000, max_chunk_s=1.0, min_chunk_s=0.1, overlap_s=0.2)
        # Push 1.5 s — emits one 1 s chunk, keeps 0.2 s overlap
        buf.push(np.zeros(1500, dtype=np.float32))
        assert buf.buffered_s == pytest.approx(0.2, abs=0.01)
