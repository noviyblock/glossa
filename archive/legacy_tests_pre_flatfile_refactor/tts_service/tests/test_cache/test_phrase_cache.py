"""Tests for InMemoryPhraseCache — LRU eviction, byte budget, stats."""

from __future__ import annotations

import pytest

from app.cache.phrase_cache import InMemoryPhraseCache, cache_key
from app.domain.entities import AudioFormat, PhraseCacheEntry

from ..conftest import make_cache_entry


@pytest.fixture
def cache() -> InMemoryPhraseCache:
    return InMemoryPhraseCache(max_entries=4, max_bytes=4 * 1024 * 1024)


# ── Basic get/set ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_returns_none_on_miss(cache):
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get_roundtrip(cache):
    entry = make_cache_entry("k1")
    await cache.set(entry)
    retrieved = await cache.get("k1")
    assert retrieved is not None
    assert retrieved.audio_bytes == entry.audio_bytes


@pytest.mark.asyncio
async def test_hit_increments_hit_count(cache):
    entry = make_cache_entry("k2")
    await cache.set(entry)
    await cache.get("k2")
    await cache.get("k2")
    retrieved = await cache.get("k2")
    assert retrieved.hits == 3


@pytest.mark.asyncio
async def test_delete_removes_entry(cache):
    entry = make_cache_entry("k3")
    await cache.set(entry)
    await cache.delete("k3")
    assert await cache.get("k3") is None


@pytest.mark.asyncio
async def test_clear_empties_cache(cache):
    for i in range(3):
        await cache.set(make_cache_entry(f"key_{i}"))
    await cache.clear()
    stats = await cache.stats()
    assert stats["entries"] == 0
    assert stats["total_bytes"] == 0


# ── LRU eviction ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lru_evicts_oldest_on_overflow(cache):
    # cache max_entries=4
    for i in range(4):
        await cache.set(make_cache_entry(f"k{i}", f"Текст {i}."))
    # k0 is the oldest; adding k4 should evict k0
    await cache.set(make_cache_entry("k4", "Новый текст."))
    assert await cache.get("k0") is None
    assert await cache.get("k4") is not None


@pytest.mark.asyncio
async def test_lru_access_prevents_eviction(cache):
    for i in range(4):
        await cache.set(make_cache_entry(f"k{i}"))
    # Access k0 → moves it to MRU position
    await cache.get("k0")
    # Add new entry → k1 (oldest) should be evicted, not k0
    await cache.set(make_cache_entry("k4"))
    assert await cache.get("k0") is not None
    assert await cache.get("k1") is None


# ── Byte budget ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_byte_budget_evicts_when_exceeded():
    # 10-byte budget; each entry has real WAV bytes (~100+)
    # Use a very small budget to force eviction
    tiny_cache = InMemoryPhraseCache(max_entries=100, max_bytes=1)
    entry = make_cache_entry("big")
    await tiny_cache.set(entry)
    # Second set should evict the first
    entry2 = make_cache_entry("big2")
    await tiny_cache.set(entry2)
    # Both may or may not be present depending on sizes,
    # but total_bytes should not greatly exceed limit after eviction attempt
    stats = await tiny_cache.stats()
    assert stats["entries"] <= 1


# ── Stats ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_hit_rate(cache):
    entry = make_cache_entry("sx")
    await cache.set(entry)
    await cache.get("sx")        # hit
    await cache.get("nonexistent")  # miss
    stats = await cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_stats_total_bytes_tracks_correctly(cache):
    entry = make_cache_entry("by")
    await cache.set(entry)
    stats = await cache.stats()
    assert stats["total_bytes"] == len(entry.audio_bytes)
    await cache.delete("by")
    stats = await cache.stats()
    assert stats["total_bytes"] == 0


# ── cache_key ─────────────────────────────────────────────────────────────────

def test_cache_key_deterministic():
    k1 = cache_key("Привет", "aidar", 24000, AudioFormat.WAV)
    k2 = cache_key("Привет", "aidar", 24000, AudioFormat.WAV)
    assert k1 == k2


def test_cache_key_differs_for_different_voices():
    k1 = cache_key("Привет", "aidar", 24000, AudioFormat.WAV)
    k2 = cache_key("Привет", "baya", 24000, AudioFormat.WAV)
    assert k1 != k2


def test_cache_key_differs_for_different_formats():
    k1 = cache_key("Привет", "aidar", 24000, AudioFormat.WAV)
    k2 = cache_key("Привет", "aidar", 24000, AudioFormat.OGG)
    assert k1 != k2


def test_cache_key_length():
    k = cache_key("text", "aidar", 24000, AudioFormat.WAV)
    assert len(k) == 24
