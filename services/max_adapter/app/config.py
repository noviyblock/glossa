from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config import BaseServiceSettings


class MAXAdapterSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="MAX_", env_file=".env", extra="ignore")

    service_name: str = "max-adapter"
    port: int = Field(default=8005, ge=1, le=65535)

    # MAX Messenger Bot
    bot_token: str = ""
    max_api_base_url: str = "https://platform-api.max.ru"
    webhook_secret: str = ""
    webhook_path: str = "/api/v1/webhook"
    # Public HTTPS URL MAX will POST events to. Set to register on startup.
    webhook_url: str = ""

    # Pipeline service URLs
    asr_url: str = "http://asr-service:8002"
    nlp_url: str = "http://nlp-service:8003"
    tts_url: str = "http://tts-service:8004"
    cv_url:  str = "http://cv-service:8001"
    pipeline_timeout: float = Field(default=30.0, gt=0.0)

    # Session store
    session_backend: Literal["memory", "redis", "tiered"] = "tiered"
    redis_url: str = "redis://redis:6379/2"
    session_ttl_s: int = Field(default=86400, ge=60)
    max_memory_sessions: int = Field(default=10_000, ge=100)

    # Rate limiting
    rate_limit_backend: Literal["memory", "redis"] = "redis"
    rate_limit_requests: int = Field(default=30, ge=1)
    rate_limit_window_s: int = Field(default=60, ge=1)

    # Event queue
    event_queue_workers: int = Field(default=4, ge=1)
    event_queue_maxsize: int = Field(default=512, ge=1)

    # Legacy SDK fields kept for health endpoint compatibility
    sdk_endpoint: str = "http://max-server:50051"
    batch_size: int = Field(default=1, ge=1)
    max_retries: int = Field(default=3, ge=0)


@lru_cache
def get_settings() -> MAXAdapterSettings:
    return MAXAdapterSettings()
