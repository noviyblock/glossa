from __future__ import annotations

from collections import OrderedDict
import io
import logging
import os
from threading import Lock
import wave

import torch

from config import (
    CACHE_SIZE,
    DEFAULT_SPEAKER,
    DEVICE,
    SAMPLE_RATE,
    SILERO_MODEL,
    SILERO_REPO,
    TORCH_HOME,
)

logger = logging.getLogger(__name__)


class _LRUCache:
    def __init__(self, maxsize: int) -> None:
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

    def set(self, key: str, value: bytes) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# Voices available in Silero v4 Russian model
_AVAILABLE_VOICES = ["aidar", "baya", "kseniya", "xenia", "random"]


class Synthesizer:
    def __init__(self) -> None:
        os.makedirs(TORCH_HOME, exist_ok=True)
        logger.info("Loading Silero TTS v4 from torch.hub…")

        self._model, _ = torch.hub.load(
            repo_or_dir=SILERO_REPO,
            model=SILERO_MODEL,
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        self._model.to(DEVICE)
        self._model.eval()
        self._cache = _LRUCache(CACHE_SIZE)
        logger.info("Silero TTS ready — sample_rate=%d device=%s", SAMPLE_RATE, DEVICE)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def voices(self) -> list[str]:
        return _AVAILABLE_VOICES

    def synthesize(self, text: str, speaker: str = DEFAULT_SPEAKER) -> bytes:
        """Return WAV bytes for the given text and speaker.

        Result is cached by (text, speaker) key.
        """
        cache_key = f"{speaker}|{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        wav = self._run_model(text, speaker)
        wav_bytes = self._to_wav_bytes(wav)
        self._cache.set(cache_key, wav_bytes)
        return wav_bytes

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_model(self, text: str, speaker: str) -> torch.Tensor:
        with torch.no_grad():
            audio = self._model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=SAMPLE_RATE,
            )
        return audio

    @staticmethod
    def _to_wav_bytes(audio: torch.Tensor) -> bytes:
        pcm = (audio.squeeze().cpu().numpy() * 32767).astype("int16")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
