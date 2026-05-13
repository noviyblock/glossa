from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class GlossaModel(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class ServiceStatus(str):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(GlossaModel):
    name: str
    status: str
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(GlossaModel):
    service: str
    status: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: list[ComponentHealth] = Field(default_factory=list)
    uptime_seconds: float | None = None


class ErrorDetail(GlossaModel):
    code: str
    message: str
    details: dict[str, object] | None = None


class ErrorResponse(GlossaModel):
    error: ErrorDetail
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
