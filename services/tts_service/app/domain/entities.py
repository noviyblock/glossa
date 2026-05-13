"""TTS domain entities — pure Python, no I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class AudioFormat(StrEnum):
    WAV = "wav"
    OGG = "ogg"
    PCM = "pcm"   # raw float32, internal use


class SynthesisStatus(StrEnum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CACHED    = "cached"


# ── Voice profile ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VoiceProfile:
    """Speaker abstraction — supports Silero today, XTTS tomorrow."""

    id: str
    display_name: str
    speaker_key: str          # engine-specific key (e.g. "aidar" for Silero)
    language: str = "ru"
    gender: str = "unknown"   # "male" | "female" | "unknown"
    engine: str = "silero"    # "silero" | "xtts" | "custom"
    sample_rate: int = 24000
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Synthesis request / result ────────────────────────────────────────────────

@dataclass
class SynthesisRequest:
    text: str
    voice_id: str = "aidar"
    language: str = "ru"
    sample_rate: int = 24000
    format: AudioFormat = AudioFormat.WAV
    put_accent: bool = True
    put_yo: bool = True
    session_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = field(default_factory=lambda: str(uuid4()))
    priority: int = 0          # lower = higher priority in queue
    chunk_size_chars: int = 150


@dataclass
class SynthesisResult:
    request_id: str
    session_id: str
    audio_bytes: bytes
    format: AudioFormat
    sample_rate: int
    duration_s: float
    inference_ms: float
    total_ms: float
    char_count: int
    from_cache: bool = False
    voice_id: str = "aidar"
    chunk_count: int = 1


# ── Streaming chunk ───────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    request_id: str
    chunk_index: int
    audio_bytes: bytes         # encoded WAV/OGG bytes for this sentence
    duration_s: float
    is_final: bool = False
    sentence_text: str = ""


# ── Queue job ─────────────────────────────────────────────────────────────────

@dataclass
class SynthesisJob:
    request: SynthesisRequest
    submitted_at: float = field(default_factory=time.perf_counter)

    def __lt__(self, other: "SynthesisJob") -> bool:
        # Lower priority value = higher priority; break ties by submission time
        if self.request.priority != other.request.priority:
            return self.request.priority < other.request.priority
        return self.submitted_at < other.submitted_at


# ── Cache entry ───────────────────────────────────────────────────────────────

@dataclass
class PhraseCacheEntry:
    key: str
    audio_bytes: bytes
    format: AudioFormat
    sample_rate: int
    duration_s: float
    char_count: int
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


# ── Benchmark ─────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    n_runs: int
    text_len: int
    voice_id: str
    sample_rate: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    rtf: float                 # real-time factor = inference_s / audio_duration_s
    chars_per_sec: float
    format: AudioFormat = AudioFormat.WAV
