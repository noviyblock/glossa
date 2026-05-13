from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

logger = get_logger(__name__)


class FasterWhisperRunner:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        model_dir: str | None = None,
        language: str = "ru",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model_dir = model_dir
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._word_timestamps = word_timestamps
        self._model: object | None = None

    def _load_model(self) -> object:
        from faster_whisper import WhisperModel

        download_root = str(Path(self._model_dir) if self._model_dir else Path("/models/whisper"))
        model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=download_root,
        )
        logger.info(
            "whisper_loaded",
            size=self._model_size,
            device=self._device,
            compute=self._compute_type,
        )
        return model

    async def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> dict:
        await self.ensure_loaded()

        with MODEL_INFERENCE_LATENCY.labels(service="asr-service", model="faster-whisper").time():
            result = await asyncio.to_thread(self._run_transcription, audio_bytes, sample_rate)
        return result

    def _run_transcription(self, audio_bytes: bytes, sample_rate: int) -> dict:
        import wave
        from faster_whisper import WhisperModel

        model: WhisperModel = self._model  # type: ignore[assignment]

        # Convert raw PCM bytes to float32 numpy array if not WAV
        if audio_bytes[:4] == b"RIFF":
            with io.BytesIO(audio_bytes) as buf:
                with wave.open(buf) as wf:
                    raw = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = sample_rate

        segments, info = model.transcribe(
            audio_np,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            word_timestamps=self._word_timestamps,
        )

        text_parts: list[str] = []
        words_out: list[dict] = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if self._word_timestamps and seg.words:
                for w in seg.words:
                    words_out.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

        return {
            "text": " ".join(text_parts),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "words": words_out,
        }

    async def transcribe_stream(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Yields partial transcription segments as they become available."""
        await self.ensure_loaded()

        from faster_whisper import WhisperModel

        model: WhisperModel = self._model  # type: ignore[assignment]
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        def _iter_segments():
            segs, info = model.transcribe(
                audio_np,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
            )
            for i, seg in enumerate(segs):
                yield {
                    "chunk_index": i,
                    "text": seg.text.strip(),
                    "confidence": float(seg.avg_logprob),
                    "no_speech_prob": float(seg.no_speech_prob),
                    "is_final": False,
                }

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def _producer():
            for item in _iter_segments():
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _producer)

        chunk_index = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            item["chunk_index"] = chunk_index
            chunk_index += 1
            yield item
