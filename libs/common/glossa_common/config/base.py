from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    service_name: str = "glossa-service"
    service_version: str = "0.1.0"

    redis_url: str = "redis://redis:6379/0"
    redis_stream_max_len: int = Field(default=10_000, ge=100)
    redis_consumer_group: str = "glossa-consumers"

    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_traces_sampler_arg: float = Field(default=1.0, ge=0.0, le=1.0)

    mlflow_tracking_uri: str = "http://mlflow:5000"

    @field_validator("otel_traces_sampler_arg", mode="before")
    @classmethod
    def parse_sampler_arg(cls, v: object) -> float:
        return float(v)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


class RedisStreamConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    redis_url: str = "redis://redis:6379/0"
    redis_stream_max_len: int = 10_000
    redis_consumer_group: str = "glossa-consumers"
    redis_block_ms: int = 1000
    redis_batch_size: int = 10
