"""Transcription REST endpoints — single-shot and language detection."""

from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...dependencies import get_medical_adapter, get_transcriber
from ...domain.entities import WordTimestamp
from ...schemas.request import AddMedicalTermRequest, DetectLanguageRequest, TranscribeRequest
from ...schemas.response import (
    DetectLanguageResponse,
    MessageResponse,
    SegmentResponse,
    TranscribeResponse,
    WordTimestampResponse,
)
from ...transcription.whisper_runner import FasterWhisperRunner

router = APIRouter(prefix="/api/v1", tags=["transcription"])


def _decode_audio(audio_b64: str, sample_rate: int) -> np.ndarray:
    try:
        raw = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")
    return FasterWhisperRunner.decode_audio(raw, sample_rate)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    body: TranscribeRequest,
    transcriber=Depends(get_transcriber),
    medical=Depends(get_medical_adapter),
) -> TranscribeResponse:
    """Single-shot transcription from base64-encoded audio."""
    audio = _decode_audio(body.audio_b64, body.sample_rate)

    t0 = time.perf_counter()
    try:
        segments = await transcriber.transcribe(
            audio,
            sample_rate=body.sample_rate,
            language=body.language,
            beam_size=body.beam_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    inference_ms = (time.perf_counter() - t0) * 1000
    text = " ".join(s.text for s in segments if s.text).strip()

    corrected, medical_terms = ("", [])
    if body.medical_adaptation:
        corrected, medical_terms = medical.adapt(text, body.language or "ru")

    all_words: list[WordTimestampResponse] = []
    seg_responses: list[SegmentResponse] = []
    for seg in segments:
        seg_responses.append(SegmentResponse(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            confidence=round(seg.confidence, 4),
            no_speech_prob=round(seg.no_speech_prob, 4),
        ))
        for w in seg.words:
            all_words.append(WordTimestampResponse(
                word=w.word, start=w.start, end=w.end, probability=w.probability
            ))

    return TranscribeResponse(
        text=text,
        corrected_text=corrected or text,
        language=body.language or "ru",
        language_probability=1.0,
        duration_s=round(len(audio) / body.sample_rate, 3),
        inference_ms=round(inference_ms, 2),
        segments=seg_responses,
        words=all_words,
        medical_terms_found=medical_terms,
    )


@router.post("/transcribe/stream")
async def transcribe_stream(
    body: TranscribeRequest,
    transcriber=Depends(get_transcriber),
    medical=Depends(get_medical_adapter),
):
    """Streaming transcription — yields NDJSON segments as they decode."""
    import json

    audio = _decode_audio(body.audio_b64, body.sample_rate)

    async def _generate():
        async for seg in transcriber.transcribe_streaming(
            audio,
            sample_rate=body.sample_rate,
            language=body.language,
            beam_size=body.beam_size,
        ):
            data = {
                "type": "segment",
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": round(seg.confidence, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "is_final": False,
            }
            yield json.dumps(data, ensure_ascii=False) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@router.post("/detect-language", response_model=DetectLanguageResponse)
async def detect_language(
    body: DetectLanguageRequest,
    transcriber=Depends(get_transcriber),
) -> DetectLanguageResponse:
    audio = _decode_audio(body.audio_b64, body.sample_rate)
    try:
        lang, prob = await transcriber.detect_language(audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DetectLanguageResponse(language=lang, probability=round(prob, 4))


@router.post("/medical-terms", response_model=MessageResponse, status_code=201)
async def add_medical_term(
    body: AddMedicalTermRequest,
    medical=Depends(get_medical_adapter),
) -> MessageResponse:
    """Register a custom medical terminology correction at runtime."""
    medical.add_term(body.term, body.correction, body.language)
    return MessageResponse(message=f"Term '{body.term}' → '{body.correction}' registered")
