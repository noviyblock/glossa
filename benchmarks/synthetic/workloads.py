"""Synthetic workload data — gesture keypoints, audio, and payloads.

All generators produce realistic data matching the actual service API contracts
(verified against services/cv_service, asr_service, nlp_service, tts_service).
"""

from __future__ import annotations

import base64
import math
import random
import struct
import time
from typing import Any


# ── RSL gloss vocabulary ──────────────────────────────────────────────────────

_RSL_GLOSSES = [
    "ПРИВЕТ", "ПОКА", "СПАСИБО", "ПОЖАЛУЙСТА", "КАК",
    "ДЕЛА", "ХОРОШО", "ПЛОХО", "ДА", "НЕТ",
    "ПОМОЩЬ", "ВРАЧ", "БОЛЬНИЦА", "ЛЕКАРСТВО", "БОЛЬ",
    "ДЕНЬГИ", "БАНК", "КАРТА", "ОПЛАТА", "СЧЁТ",
    "ИМЯ", "ТЕЛЕФОН", "АДРЕС", "ГОРОД", "ДОМ",
    "ЕСТЬ", "ПИТЬ", "ВОДА", "ЕДА", "ХЛЕБ",
    "РАБОТА", "УЧЁБА", "ШКОЛА", "УНИВЕРСИТЕТ",
    "СЕМЬЯ", "МАМА", "ПАПА", "БРАТ", "СЕСТРА",
    "ВЧЕРА", "СЕГОДНЯ", "ЗАВТРА", "ВРЕМЯ", "ЧАС",
    "ХОТЕТЬ", "ИДТИ", "ПРИХОДИТЬ", "ЗНАТЬ", "ПОНИМАТЬ",
]

_GLOSS_SEQUENCES = [
    "ПРИВЕТ КАК ДЕЛА",
    "ПОМОЩЬ НУЖЕН ВРАЧ",
    "СПАСИБО ПОМОЩЬ",
    "КАК ИМЯ",
    "СЕГОДНЯ БОЛЕТЬ ГОЛОВА",
    "НУЖЕН ЛЕКАРСТВО",
    "БАНК ГДЕ",
    "ОПЛАТА КАРТА",
    "ЕДА ХОТЕТЬ",
    "ХОРОШО ПОНИМАТЬ ДА",
    "ТЕЛЕФОН НОМЕР СКАЖИ",
    "ЗАВТРА ВСТРЕЧА ГДЕ ВРЕМЯ",
]

_TTS_TEXTS = [
    "Как вы себя чувствуете сегодня?",
    "Чем я могу вам помочь?",
    "Пожалуйста, повторите ещё раз.",
    "Я вас хорошо понимаю.",
    "Обратитесь к врачу.",
    "Ваш запрос обрабатывается.",
]


# ── Gesture keypoints ─────────────────────────────────────────────────────────

def make_pose_keypoints(
    n_landmarks: int = 33,
    noise: float = 0.02,
) -> list[list[float]]:
    """Generate realistic DWPose landmark coordinates."""
    rng = random.Random()
    landmarks = []
    for i in range(n_landmarks):
        # Approximate body landmark positions (normalized 0–1)
        if i < 11:       # Face landmarks roughly in upper portion
            x = 0.5 + rng.gauss(0, 0.1)
            y = 0.2 + rng.gauss(0, 0.05)
        elif i < 25:     # Upper body
            x = 0.5 + rng.gauss(0, 0.2)
            y = 0.5 + rng.gauss(0, 0.15)
        else:            # Lower body
            x = 0.5 + rng.gauss(0, 0.15)
            y = 0.75 + rng.gauss(0, 0.1)
        z = rng.gauss(0, 0.05)
        v = max(0.0, 1.0 - abs(rng.gauss(0, 0.3)))  # visibility
        landmarks.append([
            max(0.0, min(1.0, x + rng.gauss(0, noise))),
            max(0.0, min(1.0, y + rng.gauss(0, noise))),
            z + rng.gauss(0, noise),
            v,
        ])
    return landmarks


def make_gesture_sequence(
    seq_len: int = 30,
    n_landmarks: int = 33,
) -> list[list[list[float]]]:
    """Generate a sequence of pose frames for STGCN input."""
    return [make_pose_keypoints(n_landmarks) for _ in range(seq_len)]


def make_gesture_payload(
    seq_len: int = 30,
    n_landmarks: int = 33,
    session_id: str | None = None,
) -> dict[str, Any]:
    """POST payload for cv-service /recognize endpoint."""
    return {
        "session_id": session_id or f"bench-{int(time.time() * 1000)}",
        "frames": make_gesture_sequence(seq_len, n_landmarks),
        "timestamp": time.time(),
        "domain": "general",
    }


# ── Video frame (WebSocket) ───────────────────────────────────────────────────

def make_video_frame_bytes(width: int = 320, height: int = 240) -> bytes:
    """Generate a minimal valid JPEG-like byte sequence for WebSocket video frames."""
    # JPEG SOI + minimal headers + EOI (not a real image but correct framing)
    soi = b"\xff\xd8\xff\xe0"
    body = bytes([random.randint(0, 255) for _ in range(width * height // 8)])
    eoi = b"\xff\xd9"
    return soi + body + eoi


def make_ws_video_frame_message(session_id: str | None = None) -> dict[str, Any]:
    """WebSocket message payload for gesture_to_text session."""
    return {
        "type": "video_frame",
        "frame": base64.b64encode(make_video_frame_bytes()).decode("ascii"),
        "session_id": session_id or f"ws-bench-{int(time.time() * 1000)}",
        "timestamp": time.time(),
    }


# ── Audio ─────────────────────────────────────────────────────────────────────

def make_audio_wav(
    duration_s: float = 1.0,
    sample_rate: int = 16_000,
    frequency_hz: float = 440.0,
    noise_amplitude: float = 0.1,
) -> bytes:
    """Generate a WAV file bytes for ASR benchmarking."""
    n_samples = int(duration_s * sample_rate)
    samples: list[int] = []
    for i in range(n_samples):
        t = i / sample_rate
        signal = math.sin(2 * math.pi * frequency_hz * t)
        noise = random.gauss(0, noise_amplitude)
        val = int(max(-1.0, min(1.0, signal + noise)) * 32767)
        samples.append(val)

    # Build WAV header
    data_size = n_samples * 2  # 16-bit samples
    wav = struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE")
    wav += struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    wav += struct.pack("<4sI", b"data", data_size)
    wav += struct.pack(f"<{n_samples}h", *samples)
    return wav


def make_audio_payload(
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
    session_id: str | None = None,
) -> dict[str, Any]:
    """POST payload for asr-service /transcribe endpoint."""
    wav_bytes = make_audio_wav(duration_s, sample_rate)
    return {
        "session_id": session_id or f"bench-{int(time.time() * 1000)}",
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
        "sample_rate": sample_rate,
        "language": "ru",
        "domain": "general",
    }


def make_ws_audio_message(session_id: str | None = None) -> dict[str, Any]:
    """WebSocket message payload for speech_to_text session."""
    wav = make_audio_wav(duration_s=0.5)
    return {
        "type": "audio_chunk",
        "audio": base64.b64encode(wav).decode("ascii"),
        "session_id": session_id or f"ws-bench-{int(time.time() * 1000)}",
        "domain": "general",
        "timestamp": time.time(),
    }


# ── NLP ───────────────────────────────────────────────────────────────────────

def make_gloss_payload(
    session_id: str | None = None,
    length: str = "medium",
) -> dict[str, Any]:
    """POST payload for nlp-service /translate endpoint."""
    if length == "short":
        gloss = random.choice(_GLOSS_SEQUENCES[:4])
    elif length == "long":
        gloss = " ".join(random.choices(_RSL_GLOSSES, k=12))
    else:
        gloss = random.choice(_GLOSS_SEQUENCES)
    return {
        "session_id": session_id or f"bench-{int(time.time() * 1000)}",
        "gloss_sequence": gloss,
        "domain": random.choice(["general", "medical", "banking"]),
        "context": [],
    }


# ── TTS ───────────────────────────────────────────────────────────────────────

def make_tts_payload(session_id: str | None = None) -> dict[str, Any]:
    """POST payload for tts-service /synthesize endpoint."""
    return {
        "session_id": session_id or f"bench-{int(time.time() * 1000)}",
        "text": random.choice(_TTS_TEXTS),
        "speaker": "aidar",
        "language": "ru",
        "sample_rate": 24_000,
    }


# ── Network simulation helpers ────────────────────────────────────────────────

async def simulate_mobile_latency(rtt_ms: float, jitter_ms: float = 10.0) -> None:
    """Inject artificial network delay to simulate mobile conditions."""
    import asyncio
    import math
    delay = rtt_ms / 2.0 + abs(random.gauss(0, jitter_ms / 2.0))
    await asyncio.sleep(delay / 1000.0)


def apply_bandwidth_limit(
    data: bytes,
    bandwidth_kbps: float,
    latency_ms: float = 0.0,
) -> float:
    """Calculate how long it would take to transmit `data` on a constrained link."""
    transfer_ms = (len(data) * 8) / bandwidth_kbps  # bytes × 8 bits / (kbps)
    return latency_ms + transfer_ms
