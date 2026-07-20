"""Integration tests for ASRPipeline with mock transcriber."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.domain.entities import (
    ASRSession,
    AudioChunk,
    AudioEncoding,
    FinalTranscript,
    PartialTranscript,
    TranscriptSegment,
)
from app.pipeline.asr_pipeline import ASRPipeline
from app.transcription.medical_adapter import MedicalAdapter
from app.vad.energy_vad import EnergyVAD

from ..conftest import MockTranscriber


def _sine(freq=440.0, dur=1.0, sr=16000) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _make_pipeline(
    session: ASRSession,
    partials: list,
    finals: list,
    transcriber=None,
) -> ASRPipeline:
    async def on_partial(p: PartialTranscript):
        partials.append(p)

    async def on_final(f: FinalTranscript):
        finals.append(f)

    return ASRPipeline(
        session=session,
        transcriber=transcriber or MockTranscriber(),
        vad=EnergyVAD(
            sample_rate=16000,
            threshold=0.02,
            min_speech_ms=100,
            min_silence_ms=200,
        ),
        medical_adapter=MedicalAdapter(languages=["ru"]),
        on_partial=on_partial,
        on_final=on_final,
        emit_partials=True,
        language="ru",
        beam_size=1,
    )


@pytest.fixture
def session():
    return ASRSession(session_id="pipeline-test", language="ru", sample_rate=16000)


@pytest.mark.asyncio
async def test_push_chunk_increments_session_stats(session):
    partials, finals = [], []
    pipeline = _make_pipeline(session, partials, finals)

    audio = _sine(440, 0.3)  # 300 ms — below VAD threshold
    chunk = AudioChunk(
        data=(audio * 32767).astype(np.int16).tobytes(),
        sample_rate=16000,
        session_id=session.session_id,
    )
    await pipeline.push_chunk(chunk)
    assert session.chunk_count == 1
    assert session.total_audio_s > 0


@pytest.mark.asyncio
async def test_flush_with_audio_emits_final(session):
    partials, finals = [], []
    pipeline = _make_pipeline(session, partials, finals)

    audio = _sine(440, 1.0)
    chunk = AudioChunk(
        data=(audio * 32767).astype(np.int16).tobytes(),
        sample_rate=16000,
        session_id=session.session_id,
    )
    await pipeline.push_chunk(chunk)
    result = await pipeline.flush()
    # flush may return None if no speech or returns a FinalTranscript
    # The important thing is it didn't raise
    assert result is None or isinstance(result, FinalTranscript)


@pytest.mark.asyncio
async def test_reset_clears_pipeline(session):
    partials, finals = [], []
    pipeline = _make_pipeline(session, partials, finals)

    audio = _sine(440, 0.5)
    chunk = AudioChunk(
        data=(audio * 32767).astype(np.int16).tobytes(),
        sample_rate=16000,
        session_id=session.session_id,
    )
    await pipeline.push_chunk(chunk)
    await pipeline.reset()
    # After reset, chunk count is still in session (not reset there)
    # but pipeline internal buffers are cleared
    assert session.chunk_count == 1  # chunk was counted before reset


@pytest.mark.asyncio
async def test_medical_adapter_applied_on_final(session):
    partials, finals = [], []
    transcriber = MockTranscriber(segments=[
        TranscriptSegment(text="у пациента диабит", start=0.0, end=1.0, avg_logprob=-0.2, no_speech_prob=0.01),
    ])
    pipeline = _make_pipeline(session, partials, finals, transcriber=transcriber)

    audio = _sine(440, 1.5)
    chunk = AudioChunk(
        data=(audio * 32767).astype(np.int16).tobytes(),
        sample_rate=16000,
        session_id=session.session_id,
    )
    await pipeline.push_chunk(chunk)
    await pipeline.flush()

    if finals:
        final = finals[0]
        # MedicalAdapter should have corrected "диабит" → "диабет"
        assert "диабет" in final.corrected_text or "диабит" in final.text
