"""ASR domain entities — pure Python, zero I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

import numpy as np


class TranscriptType(StrEnum):
    PARTIAL = "partial"
    FINAL   = "final"


class AudioEncoding(StrEnum):
    PCM_S16LE = "pcm_s16le"   # raw 16-bit signed little-endian
    WAV       = "wav"
    WEBM_OPUS = "webm_opus"
    OGG_OPUS  = "ogg_opus"


class ASRSessionStatus(StrEnum):
    ACTIVE       = "active"
    DISCONNECTED = "disconnected"
    CLOSED       = "closed"


class VADState(StrEnum):
    SPEECH  = "speech"
    SILENCE = "silence"


# ── Audio primitives ──────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    """Single raw audio chunk as received from the client."""
    chunk_id: str       = field(default_factory=lambda: str(uuid4()))
    data: bytes         = field(default=b"")
    sample_rate: int    = 16000
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    session_id: str     = ""
    seq: int            = 0
    received_at: float  = field(default_factory=time.perf_counter)

    @property
    def duration_s(self) -> float:
        n_samples = len(self.data) // 2  # int16 = 2 bytes
        return n_samples / max(self.sample_rate, 1)

    def to_float32(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass
class VADSegment:
    """Voice-activity detected speech segment."""
    start_ms: float
    end_ms: float
    audio: np.ndarray           # float32 samples
    sample_rate: int = 16000
    state: VADState = VADState.SPEECH
    confidence: float = 1.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


# ── Transcript types ──────────────────────────────────────────────────────────

@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds
    end: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """One whisper segment — a phrase or sentence."""
    segment_id: str     = field(default_factory=lambda: str(uuid4()))
    text: str           = ""
    start: float        = 0.0   # seconds
    end: float          = 0.0
    avg_logprob: float  = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    words: list[WordTimestamp] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Map avg_logprob → [0, 1] probability."""
        import math
        return max(0.0, min(1.0, math.exp(self.avg_logprob)))


@dataclass
class PartialTranscript:
    """Intermediate transcript emitted while speech is still ongoing."""
    session_id: str
    request_id: str
    text: str
    language: str = "ru"
    language_probability: float = 1.0
    segments: list[TranscriptSegment] = field(default_factory=list)
    transcript_type: TranscriptType = TranscriptType.PARTIAL
    emitted_at: float = field(default_factory=time.perf_counter)
    chunk_seq: int = 0


@dataclass
class FinalTranscript:
    """Complete transcript for one utterance (after VAD silence or flush)."""
    session_id: str
    request_id: str
    text: str
    language: str = "ru"
    language_probability: float = 1.0
    segments: list[TranscriptSegment] = field(default_factory=list)
    words: list[WordTimestamp] = field(default_factory=list)
    duration_s: float = 0.0
    audio_duration_s: float = 0.0
    transcript_type: TranscriptType = TranscriptType.FINAL

    # Latency telemetry
    first_chunk_at: float = 0.0
    transcribed_at: float = field(default_factory=time.perf_counter)
    inference_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Medical post-processing
    medical_terms_found: list[str] = field(default_factory=list)
    corrected_text: str = ""

    @property
    def effective_text(self) -> str:
        return self.corrected_text or self.text


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class ASRSession:
    session_id: str     = field(default_factory=lambda: str(uuid4()))
    reconnect_token: str = field(default_factory=lambda: str(uuid4()))
    status: ASRSessionStatus = ASRSessionStatus.ACTIVE
    language: str       = "ru"
    sample_rate: int    = 16000
    created_at: float   = field(default_factory=time.time)
    last_active: float  = field(default_factory=time.time)
    chunk_count: int    = 0
    transcript_count: int = 0
    total_audio_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_active = time.time()


# ── Benchmark ─────────────────────────────────────────────────────────────────

@dataclass
class ASRBenchmarkResult:
    model_size: str
    device: str
    compute_type: str
    n_samples: int
    audio_duration_s: float
    mean_inference_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rtf: float          # Real-Time Factor = inference_time / audio_duration
    word_error_rate: float | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "n_samples": self.n_samples,
            "audio_duration_s": self.audio_duration_s,
            "mean_inference_ms": self.mean_inference_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "rtf": round(self.rtf, 4),
        }
