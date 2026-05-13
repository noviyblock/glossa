from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config import BaseServiceSettings


class CVServiceSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="CV_", env_file=".env", extra="ignore")

    service_name: str = "cv-service"
    port: int = Field(default=8001, ge=1, le=65535)

    model_backend: Literal["onnx", "openvino"] = "onnx"
    model_path: str = "/models/gesture_classifier.onnx"
    device: str = "CPU"  # CPU | GPU | AUTO (OpenVINO) | CUDAExecutionProvider (ONNX)
    max_batch_size: int = Field(default=4, ge=1)
    mediapipe_model_complexity: Literal[0, 1, 2] = 1
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Sliding window for sequence classification
    sequence_length: int = Field(default=30, ge=10)
    sequence_stride: int = Field(default=15, ge=1)


@lru_cache
def get_settings() -> CVServiceSettings:
    return CVServiceSettings()
