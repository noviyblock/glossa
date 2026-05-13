"""FastAPI dependency resolvers."""

from __future__ import annotations

from fastapi import Request

from .benchmark.benchmark_service import ASRBenchmarkService
from .domain.interfaces import MedicalAdapterPort, SessionStorePort, TranscriberPort, VADPort
from .queue.audio_queue import AudioChunkQueue
from .session.asr_session import InMemorySessionStore


def get_transcriber(request: Request) -> TranscriberPort:
    return request.app.state.transcriber


def get_vad(request: Request) -> VADPort:
    return request.app.state.vad


def get_medical_adapter(request: Request) -> MedicalAdapterPort:
    return request.app.state.medical_adapter


def get_audio_queue(request: Request) -> AudioChunkQueue:
    return request.app.state.audio_queue


def get_session_store(request: Request) -> InMemorySessionStore:
    return request.app.state.session_store


def get_benchmark_service(request: Request) -> ASRBenchmarkService:
    return request.app.state.benchmark_service
