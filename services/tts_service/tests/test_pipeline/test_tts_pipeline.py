"""Integration tests for TTSPipeline — cache hit/miss, streaming, preload."""

from __future__ import annotations

import pytest

from app.audio.encoder import AudioEncoder
from app.cache.phrase_cache import InMemoryPhraseCache, TieredPhraseCache, RedisPhraseCache
from app.domain.entities import AudioFormat, SynthesisRequest
from app.domain.text_processor import TextPreprocessor
from app.pipeline.tts_pipeline import TTSPipeline
from app.queue.synthesis_queue import SynthesisQueue

from ..conftest import FakeSynthesizer, make_cache_entry


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _NoopRedis(RedisPhraseCache):
    """Redis cache that does nothing — avoids real Redis in tests."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def get(self, key):
        return None

    async def set(self, entry) -> None:
        pass

    async def delete(self, key) -> None:
        pass

    async def clear(self) -> None:
        pass

    async def stats(self) -> dict:
        return {"backend": "redis", "entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "available": False}


@pytest.fixture
async def pipeline():
    synth = FakeSynthesizer()
    l1 = InMemoryPhraseCache(max_entries=64, max_bytes=32 * 1024 * 1024)
    l2 = _NoopRedis()
    cache = TieredPhraseCache(l1=l1, l2=l2)
    await cache.start()

    encoder = AudioEncoder()
    queue = SynthesisQueue(synthesizer=synth, workers=1, maxsize=16)
    await queue.start()

    pl = TTSPipeline(
        synthesizer=synth,
        cache=cache,
        encoder=encoder,
        queue=queue,
        preprocessor=TextPreprocessor(),
    )
    yield pl, synth, l1

    await queue.stop()
    await cache.stop()


# ── Full synthesis ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_returns_result(pipeline):
    pl, synth, _ = pipeline
    req = SynthesisRequest(text="Добрый день.", voice_id="aidar")
    result = await pl.synthesize(req)
    assert result.audio_bytes[:4] == b"RIFF"  # valid WAV
    assert result.duration_s > 0
    assert result.from_cache is False
    assert len(synth.calls) == 1


@pytest.mark.asyncio
async def test_synthesize_cache_hit_skips_synthesis(pipeline):
    pl, synth, l1 = pipeline
    req = SynthesisRequest(text="Добрый день.", voice_id="aidar")

    # First call → synthesizes, caches
    r1 = await pl.synthesize(req)
    assert r1.from_cache is False

    # Second call → cache hit
    r2 = await pl.synthesize(req)
    assert r2.from_cache is True
    assert len(synth.calls) == 1  # no second synthesis


@pytest.mark.asyncio
async def test_synthesize_different_voices_have_separate_cache_entries(pipeline):
    pl, synth, _ = pipeline
    req_a = SynthesisRequest(text="Привет.", voice_id="aidar")
    req_b = SynthesisRequest(text="Привет.", voice_id="baya")

    await pl.synthesize(req_a)
    await pl.synthesize(req_b)
    # Both should be synthesized (different cache keys)
    assert len(synth.calls) == 2


@pytest.mark.asyncio
async def test_synthesize_ogg_format(pipeline):
    pl, _, _ = pipeline
    req = SynthesisRequest(text="Тест.", voice_id="aidar", format=AudioFormat.OGG)
    # OGG falls back to WAV when soundfile unavailable — should not raise
    result = await pl.synthesize(req)
    assert len(result.audio_bytes) > 0


# ── Streaming synthesis ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_streaming_yields_chunks(pipeline):
    pl, synth, _ = pipeline
    req = SynthesisRequest(text="Добрый день. Как дела? Хорошо!")
    chunks = []
    async for chunk in pl.synthesize_streaming(req):
        chunks.append(chunk)

    assert len(chunks) >= 1
    assert chunks[-1].is_final is True
    assert all(len(c.audio_bytes) > 0 for c in chunks)


@pytest.mark.asyncio
async def test_synthesize_streaming_chunk_indices_sequential(pipeline):
    pl, _, _ = pipeline
    req = SynthesisRequest(text="Раз. Два. Три.")
    indices = []
    async for chunk in pl.synthesize_streaming(req):
        indices.append(chunk.chunk_index)
    assert indices == list(range(len(indices)))


@pytest.mark.asyncio
async def test_synthesize_streaming_uses_cache_on_repeated_call(pipeline):
    pl, synth, _ = pipeline
    req = SynthesisRequest(text="Добрый день. Как дела?")

    # First pass — synthesize and cache each sentence
    async for _ in pl.synthesize_streaming(req):
        pass
    first_call_count = len(synth.calls)

    # Second pass — should be served from cache
    async for _ in pl.synthesize_streaming(req):
        pass
    assert len(synth.calls) == first_call_count  # no new synthesis calls


# ── Preload phrases ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preload_phrases_warms_cache(pipeline):
    pl, synth, l1 = pipeline
    phrases = ["Привет.", "Пока.", "Спасибо."]
    loaded = await pl.preload_phrases(phrases=phrases, voice_id="aidar")
    assert loaded == 3
    stats = await l1.stats()
    assert stats["entries"] == 3


@pytest.mark.asyncio
async def test_preload_phrases_skips_already_cached(pipeline):
    pl, synth, _ = pipeline
    phrases = ["Привет."]
    await pl.preload_phrases(phrases=phrases)
    first_count = len(synth.calls)
    await pl.preload_phrases(phrases=phrases)
    # No new synthesis calls — already cached
    assert len(synth.calls) == first_count


@pytest.mark.asyncio
async def test_preload_uses_builtin_defaults_when_empty(pipeline):
    pl, synth, _ = pipeline
    loaded = await pl.preload_phrases(phrases=None)
    assert loaded > 0
    assert len(synth.calls) == loaded


# ── Benchmark ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_benchmark_returns_valid_result(pipeline):
    pl, _, _ = pipeline
    result = await pl.benchmark(text="Тест.", voice_id="aidar", n=3)
    assert result.n_runs == 3
    assert result.p50_ms >= 0
    assert result.rtf >= 0
