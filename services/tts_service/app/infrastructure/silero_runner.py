from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

logger = get_logger(__name__)


class SileroTTSRunner:
    def __init__(
        self,
        speaker: str = "aidar",
        language: str = "ru",
        sample_rate: int = 24000,
        device: str = "cpu",
        put_accent: bool = True,
        put_yo: bool = True,
        model_dir: str | None = None,
    ) -> None:
        self._speaker = speaker
        self._language = language
        self._sample_rate = sample_rate
        self._device = device
        self._put_accent = put_accent
        self._put_yo = put_yo
        self._model_dir = Path(model_dir) if model_dir else Path("/models/silero")
        self._model: object | None = None

    def _load_model(self) -> object:
        import torch

        device = torch.device(self._device)
        local_path = self._model_dir / f"silero_tts_{self._language}.pt"

        if local_path.exists():
            model = torch.package.PackageImporter(str(local_path)).load_pickle("tts_models", "model")
        else:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=self._language,
                speaker=f"v4_{self._language}",
            )
            self._model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model, str(local_path))

        model.to(device)
        logger.info(
            "silero_loaded",
            speaker=self._speaker,
            language=self._language,
            device=self._device,
        )
        return model

    async def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)

    def _synthesize_sync(self, text: str) -> tuple[bytes, float]:
        import torch

        model = self._model
        device = torch.device(self._device)

        audio = model.apply_tts(  # type: ignore[union-attr]
            text=text,
            speaker=self._speaker,
            sample_rate=self._sample_rate,
            put_accent=self._put_accent,
            put_yo=self._put_yo,
        )
        audio_np: np.ndarray = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
        duration = len(audio_np) / self._sample_rate

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())

        return buf.getvalue(), duration

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        await self.ensure_loaded()

        with MODEL_INFERENCE_LATENCY.labels(service="tts-service", model="silero").time():
            audio_bytes, duration = await asyncio.to_thread(self._synthesize_sync, text)

        logger.debug("tts_synthesized", chars=len(text), duration_s=round(duration, 2))
        return audio_bytes, duration

    async def synthesize_streaming(self, text: str, chunk_size_chars: int = 200):
        """Chunk text at sentence boundaries and stream audio chunks."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return

        buffer = ""
        for sent in sentences:
            buffer += (" " if buffer else "") + sent
            if len(buffer) >= chunk_size_chars:
                audio_bytes, _ = await self.synthesize(buffer)
                yield audio_bytes
                buffer = ""

        if buffer:
            audio_bytes, _ = await self.synthesize(buffer)
            yield audio_bytes
