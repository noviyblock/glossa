"""Audio buffering strategy — VAD-aware ring buffer with chunking."""

from __future__ import annotations

import numpy as np

from .interfaces import AudioBufferPort


class RingAudioBuffer(AudioBufferPort):
    """Fixed-capacity ring buffer for float32 PCM audio.

    Strategy:
    - Accept chunks of any size
    - Emit when buffer exceeds emit_threshold_s seconds
    - Keep a look-back of overlap_s seconds for cross-chunk context
    - Evict oldest samples when capacity is exceeded
    """

    def __init__(
        self,
        capacity_s: float = 30.0,
        sample_rate: int = 16000,
        overlap_s: float = 0.5,
        emit_threshold_s: float = 1.0,
    ) -> None:
        self._sr = sample_rate
        self._cap = int(capacity_s * sample_rate)
        self._overlap = int(overlap_s * sample_rate)
        self._threshold = int(emit_threshold_s * sample_rate)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._write = 0      # write cursor (absolute position)
        self._n = 0          # number of valid samples

    # ── AudioBufferPort ───────────────────────────────────────────────────────

    def push(self, audio: np.ndarray) -> None:
        audio = audio.ravel().astype(np.float32)
        n = len(audio)
        if n == 0:
            return

        if n >= self._cap:
            # Incoming chunk larger than buffer — keep only the tail
            audio = audio[-self._cap:]
            n = self._cap

        start = self._write % self._cap
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = audio
        else:
            first = self._cap - start
            self._buf[start:] = audio[:first]
            self._buf[:end - self._cap] = audio[first:]

        self._write += n
        self._n = min(self._n + n, self._cap)

    def flush(self) -> np.ndarray:
        """Return all buffered samples and reset (preserving overlap)."""
        out = self._read_all()
        # Keep overlap for cross-boundary context
        if len(out) > self._overlap:
            self._reset_with(out[-self._overlap:])
        else:
            self._reset_with(out)
        return out

    def peek(self, n_samples: int) -> np.ndarray:
        """Return last n_samples without consuming them."""
        n_samples = min(n_samples, self._n)
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)
        end = self._write % self._cap
        start = (end - n_samples) % self._cap
        if start < end:
            return self._buf[start:end].copy()
        return np.concatenate([self._buf[start:], self._buf[:end]])

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._write = 0
        self._n = 0

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def duration_s(self) -> float:
        return self._n / self._sr

    @property
    def ready_to_emit(self) -> bool:
        return self._n >= self._threshold

    # ── Internals ─────────────────────────────────────────────────────────────

    def _read_all(self) -> np.ndarray:
        if self._n == 0:
            return np.zeros(0, dtype=np.float32)
        end = self._write % self._cap
        start = (end - self._n) % self._cap
        if start < end:
            return self._buf[start:end].copy()
        return np.concatenate([self._buf[start:], self._buf[:end]])

    def _reset_with(self, audio: np.ndarray) -> None:
        self._buf[:] = 0.0
        n = min(len(audio), self._cap)
        self._buf[:n] = audio[-n:]
        self._write = n
        self._n = n


class ChunkedAudioBuffer:
    """Higher-level buffer that emits fixed-size chunks for Whisper.

    Whisper works best on 30-second windows. This buffer accumulates audio
    and emits (context_audio, new_audio) pairs for streaming inference.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_chunk_s: float = 30.0,
        min_chunk_s: float = 0.5,
        overlap_s: float = 2.0,
    ) -> None:
        self._sr = sample_rate
        self._max_samples = int(max_chunk_s * sample_rate)
        self._min_samples = int(min_chunk_s * sample_rate)
        self._overlap = int(overlap_s * sample_rate)
        self._accumulated: list[np.ndarray] = []
        self._total = 0

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        """Push audio samples. Returns list of chunks ready for transcription."""
        self._accumulated.append(audio.ravel().astype(np.float32))
        self._total += len(audio)

        chunks_out: list[np.ndarray] = []
        while self._total >= self._max_samples:
            combined = np.concatenate(self._accumulated)
            chunk = combined[: self._max_samples]
            chunks_out.append(chunk)
            # Keep overlap for next window
            remainder = combined[self._max_samples - self._overlap :]
            self._accumulated = [remainder]
            self._total = len(remainder)

        return chunks_out

    def flush(self) -> np.ndarray | None:
        """Flush remaining audio (below max_chunk_s). Returns None if too short."""
        if self._total < self._min_samples:
            return None
        combined = np.concatenate(self._accumulated) if self._accumulated else np.zeros(0, dtype=np.float32)
        self.reset()
        return combined

    def reset(self) -> None:
        self._accumulated = []
        self._total = 0

    @property
    def buffered_s(self) -> float:
        return self._total / self._sr
