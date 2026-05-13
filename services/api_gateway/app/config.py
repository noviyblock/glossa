from functools import lru_cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config import BaseServiceSettings


class GatewaySettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    service_name: str = "api-gateway"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    secret_key: str = "change-me-in-production"
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit_rpm: int = Field(default=60, ge=1)

    cv_service_url: str = "http://cv-service:8001"
    asr_service_url: str = "http://asr-service:8002"
    nlp_service_url: str = "http://nlp-service:8003"
    tts_service_url: str = "http://tts-service:8004"
    max_adapter_url: str = "http://max-adapter:8005"

    http_timeout: float = Field(default=30.0, gt=0.0)
    http_max_connections: int = Field(default=100, ge=10)


@lru_cache
def get_settings() -> GatewaySettings:
    return GatewaySettings()
