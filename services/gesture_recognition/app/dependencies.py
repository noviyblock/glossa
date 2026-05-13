"""FastAPI dependency injection resolvers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .registry.model_registry import ModelRegistry
from .service.benchmark_service import BenchmarkService
from .service.recognition_service import GestureRecognitionService


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_recognition_service(request: Request) -> GestureRecognitionService:
    return request.app.state.recognition_service


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service
