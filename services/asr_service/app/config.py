"""ASR service configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config.base import BaseServiceSettings


class ASRSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    service_name: str = "asr-service"
    port: int = Field(default=8002, ge=1, le=65535)

    # Whisper model
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "base"
    device: str = "auto"           # auto | cpu | cuda
    compute_type: str = "auto"     # auto | int8 | float16 | float32 | int8_float16
    model_dir: Path = Path("/models/whisper")
    num_workers: int = Field(default=1, ge=1, le=8)
    cpu_threads: int = Field(default=4, ge=1, le=32)

    # Transcription defaults
    language: str = "ru"
    beam_size: int = Field(default=5, ge=1, le=20)
    best_of: int = Field(default=5, ge=1, le=20)
    word_timestamps: bool = False
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    temperature: float = 0.0

    # VAD
    vad_backend: Literal["silero", "energy"] = "silero"
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_speech_ms: int = Field(default=250, ge=50)
    vad_min_silence_ms: int = Field(default=500, ge=100)
    vad_model_path: Path | None = None

    # Streaming
    emit_partials: bool = True
    partial_min_delta_chars: int = 3

    # Audio queue
    audio_queue_maxsize: int = Field(default=256, ge=8)
    audio_queue_workers: int = Field(default=2, ge=1)

    # Session
    session_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://redis:6379/1"

    # Buffer
    ring_buffer_capacity_s: float = 60.0
    ring_buffer_emit_threshold_s: float = 0.5
    chunk_buffer_max_s: float = 30.0
    chunk_buffer_overlap_s: float = 2.0

    # Medical adapter
    medical_adaptation: bool = True
    medical_languages: list[str] = ["ru", "en"]

    # Warmup
    warmup_iterations: int = 3


@lru_cache
def get_settings() -> ASRSettings:
    return ASRSettings()
