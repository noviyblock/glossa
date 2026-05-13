"""Silero VAD wrapper — neural VAD using the silero-vad ONNX model.

Falls back to EnergyVAD if torch / silero_vad not installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import VADSegment, VADState
from ..domain.interfaces import VADPort

logger = get_logger(__name__)

_SILERO_REPO = "snakers4/silero-vad"
_SILERO_MODEL = "silero_vad"


class SileroVAD(VADPort):
    """Silero VAD — state-of-the-art lightweight neural VAD.

    Runs synchronously (model is tiny, <1 ms per frame).
    All async entry points offload to a thread.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        window_size_samples: int = 512,
        speech_pad_ms: int = 100,
        model_path: Path | None = None,
    ) -> None:
        self._sr = sample_rate
        self._threshold = threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._window = window_size_samples
        self._pad_ms = speech_pad_ms
        self._model_path = model_path
        self._model = None
        self._utils = None
        self._loaded = False

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import torch

            if self._model_path and self._model_path.exists():
                model, utils = torch.hub.load(
                    repo_or_dir=str(self._model_path.parent),
                    model=self._model_path.stem,
                    source="local",
                )
            else:
                model, utils = torch.hub.load(
                    repo_or_dir=_SILERO_REPO,
                    model=_SILERO_MODEL,
                    force_reload=False,
                )
            model.eval()
            self._model = model
            self._utils = utils
            self._loaded = True
            logger.info("silero_vad_loaded", threshold=self._threshold)
        except Exception as exc:
            logger.warning("silero_vad_load_failed", error=str(exc))
            self._loaded = False

    def process(self, audio: np.ndarray, sample_rate: int) -> list[VADSegment]:
        if not self._loaded or self._model is None:
            from .energy_vad import EnergyVAD
            return EnergyVAD(sample_rate=sample_rate).process(audio, sample_rate)

        return self._run_silero(audio, sample_rate)

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_silero(self, audio: np.ndarray, sample_rate: int) -> list[VADSegment]:
        try:
            import torch
            get_speech_timestamps = self._utils[0]

            audio_t = torch.from_numpy(audio.ravel().astype(np.float32))
            timestamps = get_speech_timestamps(
                audio_t,
                self._model,
                sampling_rate=sample_rate,
                threshold=self._threshold,
                min_speech_duration_ms=self._min_speech_ms,
                min_silence_duration_ms=self._min_silence_ms,
                speech_pad_ms=self._pad_ms,
                return_seconds=False,
            )
            segments: list[VADSegment] = []
            for ts in timestamps:
                start = int(ts["start"])
                end = int(ts["end"])
                seg_audio = audio[start:end]
                segments.append(VADSegment(
                    start_ms=start / sample_rate * 1000,
                    end_ms=end / sample_rate * 1000,
                    audio=seg_audio,
                    sample_rate=sample_rate,
                    state=VADState.SPEECH,
                    confidence=1.0,
                ))
            return segments
        except Exception as exc:
            logger.warning("silero_vad_inference_error", error=str(exc))
            from .energy_vad import EnergyVAD
            return EnergyVAD(sample_rate=sample_rate).process(audio, sample_rate)


def create_vad(
    backend: str = "silero",
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    model_path: Path | None = None,
) -> VADPort:
    """Factory — returns Silero if available, else EnergyVAD."""
    if backend == "energy":
        from .energy_vad import EnergyVAD
        return EnergyVAD(sample_rate=sample_rate, min_speech_ms=min_speech_ms, min_silence_ms=min_silence_ms)

    vad = SileroVAD(
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        model_path=model_path,
    )
    return vad
