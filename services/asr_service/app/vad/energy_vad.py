"""Energy-based VAD — simple RMS threshold, no dependencies.

Used as fallback when Silero VAD is unavailable.
"""

from __future__ import annotations

import numpy as np

from ..domain.entities import VADSegment, VADState
from ..domain.interfaces import VADPort


class EnergyVAD(VADPort):
    """Short-time RMS energy threshold VAD.

    Splits audio into frames and labels each as speech / silence.
    Merges consecutive speech frames into VADSegments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        threshold: float = 0.01,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        pre_speech_ms: int = 100,
    ) -> None:
        self._sr = sample_rate
        self._frame_size = int(sample_rate * frame_ms / 1000)
        self._threshold = threshold
        self._min_speech = int(sample_rate * min_speech_ms / 1000)
        self._min_silence = int(sample_rate * min_silence_ms / 1000)
        self._pre_speech = int(sample_rate * pre_speech_ms / 1000)

    def process(self, audio: np.ndarray, sample_rate: int) -> list[VADSegment]:
        audio = audio.ravel().astype(np.float32)
        if len(audio) == 0:
            return []

        frames = self._frame_rms(audio)
        speech_mask = frames > self._threshold
        segments = self._merge(audio, speech_mask, sample_rate)
        return segments

    def reset(self) -> None:
        pass  # stateless

    # ── Internals ─────────────────────────────────────────────────────────────

    def _frame_rms(self, audio: np.ndarray) -> np.ndarray:
        n_frames = len(audio) // self._frame_size
        if n_frames == 0:
            return np.array([np.sqrt(np.mean(audio ** 2))])
        frames = audio[: n_frames * self._frame_size].reshape(n_frames, self._frame_size)
        return np.sqrt(np.mean(frames ** 2, axis=1))

    def _merge(
        self, audio: np.ndarray, speech_mask: np.ndarray, sr: int
    ) -> list[VADSegment]:
        segments: list[VADSegment] = []
        in_speech = False
        seg_start = 0
        silence_count = 0

        for i, is_speech in enumerate(speech_mask):
            sample_idx = i * self._frame_size

            if is_speech:
                silence_count = 0
                if not in_speech:
                    # Start of speech — include pre-speech context
                    seg_start = max(0, sample_idx - self._pre_speech)
                    in_speech = True
            else:
                if in_speech:
                    silence_count += self._frame_size
                    if silence_count >= self._min_silence:
                        seg_end = sample_idx - silence_count + self._frame_size
                        if seg_end - seg_start >= self._min_speech:
                            seg_audio = audio[seg_start:seg_end]
                            segments.append(VADSegment(
                                start_ms=seg_start / sr * 1000,
                                end_ms=seg_end / sr * 1000,
                                audio=seg_audio,
                                sample_rate=sr,
                                state=VADState.SPEECH,
                                confidence=float(np.mean(seg_audio ** 2) ** 0.5 / (self._threshold + 1e-6)),
                            ))
                        in_speech = False
                        silence_count = 0

        # Flush open segment
        if in_speech:
            seg_end = len(audio)
            if seg_end - seg_start >= self._min_speech:
                seg_audio = audio[seg_start:seg_end]
                segments.append(VADSegment(
                    start_ms=seg_start / sr * 1000,
                    end_ms=seg_end / sr * 1000,
                    audio=seg_audio,
                    sample_rate=sr,
                    state=VADState.SPEECH,
                ))

        return segments
