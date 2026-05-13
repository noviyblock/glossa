"""TTS service configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config import BaseServiceSettings


class TTSSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    service_name: str = "tts-service"
    port: int = Field(default=8004, ge=1, le=65535)

    # Silero model
    model_dir: str = "/models/silero"
    language: str = "ru"
    device: str = "cpu"              # cpu | cuda
    cpu_threads: int = Field(default=4, ge=1, le=32)

    # Default voice
    default_voice_id: str = "aidar"
    default_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    default_format: Literal["wav", "ogg"] = "wav"
    put_accent: bool = True
    put_yo: bool = True

    # Warmup
    warmup_on_load: bool = True
    preload_phrases_on_start: bool = True

    # Cache (L1)
    cache_max_entries: int = Field(default=512, ge=10)
    cache_max_bytes_mb: int = Field(default=256, ge=16)  # MB

    # Cache (L2 Redis)
    cache_backend: Literal["memory", "redis", "tiered"] = "tiered"
    cache_redis_url: str = "redis://redis:6379/2"
    cache_ttl_seconds: int = Field(default=86400, ge=60)   # 24 hours

    # Synthesis queue
    queue_workers: int = Field(default=2, ge=1, le=8)
    queue_maxsize: int = Field(default=64, ge=8)
    queue_job_timeout_s: float = Field(default=30.0, ge=5.0)

    @property
    def cache_max_bytes(self) -> int:
        return self.cache_max_bytes_mb * 1024 * 1024


@lru_cache
def get_settings() -> TTSSettings:
    return TTSSettings()
