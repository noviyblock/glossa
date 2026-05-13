"""FastAPI dependency resolvers."""

from __future__ import annotations

from fastapi import Request

from .pipeline.tts_pipeline import TTSPipeline


def get_pipeline(request: Request) -> TTSPipeline:
    return request.app.state.pipeline
