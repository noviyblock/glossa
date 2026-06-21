from __future__ import annotations

from collections.abc import Iterator
import io
import logging
import wave

from faster_whisper import WhisperModel
import numpy as np

from config import BEAM_SIZE, COMPUTE_TYPE, DEVICE, LANGUAGE, VAD_FILTER, WHISPER_MODEL

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000  # Whisper always expects 16 kHz mono float32


class Transcriber:
    def __init__(self) -> None:
        logger.info(
            "Loading WhisperModel model=%s device=%s compute_type=%s",
            WHISPER_MODEL, DEVICE, COMPUTE_TYPE,
        )
        self._model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        logger.info("WhisperModel ready")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def transcribe_bytes(self, audio_bytes: bytes) -> dict:
        """Transcribe raw WAV/PCM bytes → {"text", "language", "duration_ms"}."""
        pcm = self._decode_audio(audio_bytes)
        duration_ms = int(len(pcm) / _SAMPLE_RATE * 1000)

        segments, _info = self._model.transcribe(
            pcm,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": LANGUAGE, "duration_ms": duration_ms}

    def transcribe_stream(self, audio_bytes: bytes) -> Iterator[str]:
        """Yield partial transcription strings as segments arrive."""
        pcm = self._decode_audio(audio_bytes)
        segments, _info = self._model.transcribe(
            pcm,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
        )
        for seg in segments:
            yield seg.text.strip()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_audio(data: bytes) -> np.ndarray:
        """Return float32 mono 16 kHz array from WAV bytes or raw PCM int16."""
        try:
            with wave.open(io.BytesIO(data)) as wf:
                n_channels = wf.getnchannels()
                sampwidth  = wf.getsampwidth()
                framerate  = wf.getframerate()
                raw        = wf.readframes(wf.getnframes())

            dtype = np.int16 if sampwidth == 2 else np.int32
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            if framerate != _SAMPLE_RATE:
                # Simple linear resample (good enough for diploma demo)
                ratio = _SAMPLE_RATE / framerate
                target_len = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, target_len),
                    np.arange(len(samples)),
                    samples,
                )

            # Normalise to [-1, 1]
            if sampwidth == 2:
                samples /= 32768.0
            elif sampwidth == 4:
                samples /= 2_147_483_648.0

            return samples.astype(np.float32)

        except wave.Error:
            # Treat as raw int16 PCM at 16 kHz
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
