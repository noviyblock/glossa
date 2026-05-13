"""Tiered phrase audio cache — LRU in-memory + Redis.

Architecture:
  TieredPhraseCache
    ├── InMemoryPhraseCache  (L1: fast, bounded LRU)
    └── RedisPhraseCache     (L2: persistent, TTL-based)

Cache key = SHA-256[:16] of "{text}|{voice_id}|{sample_rate}|{format}"

On GET:
  1. Check L1 → hit: return + update hit count
  2. Check L2 → hit: promote to L1, return
  3. Miss → caller synthesizes → SET both levels

On SET:
  Write L1 (evict LRU if full) + write L2 with TTL

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict

from glossa_common.logging import get_logger

from ..domain.entities import AudioFormat, PhraseCacheEntry
from ..domain.interfaces import CachePort

logger = get_logger(__name__)


def cache_key(text: str, voice_id: str, sample_rate: int, fmt: AudioFormat) -> str:
    raw = f"{text}|{voice_id}|{sample_rate}|{fmt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── L1: In-memory LRU ─────────────────────────────────────────────────────────

class InMemoryPhraseCache(CachePort):
    """Bounded LRU cache — evicts least-recently-used entry when full."""

    def __init__(self, max_entries: int = 512, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._store: OrderedDict[str, PhraseCacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._total_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0

    async def start(self) -> None:
        pass  # No background tasks needed

    async def stop(self) -> None:
        pass

    async def get(self, key: str) -> PhraseCacheEntry | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry

    async def set(self, entry: PhraseCacheEntry) -> None:
        async with self._lock:
            key = entry.key
            size = len(entry.audio_bytes)

            # If key exists, update in place
            if key in self._store:
                old = self._store[key]
                self._total_bytes -= len(old.audio_bytes)
                self._store[key] = entry
                self._store.move_to_end(key)
                self._total_bytes += size
                return

            # Evict by count
            while len(self._store) >= self._max_entries:
                _, evicted = self._store.popitem(last=False)
                self._total_bytes -= len(evicted.audio_bytes)

            # Evict by byte budget
            while self._total_bytes + size > self._max_bytes and self._store:
                _, evicted = self._store.popitem(last=False)
                self._total_bytes -= len(evicted.audio_bytes)

            self._store[key] = entry
            self._total_bytes += size

    async def delete(self, key: str) -> None:
        async with self._lock:
            entry = self._store.pop(key, None)
            if entry:
                self._total_bytes -= len(entry.audio_bytes)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._total_bytes = 0
            self._hits = 0
            self._misses = 0

    async def stats(self) -> dict:
        async with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


# ── L2: Redis-backed ──────────────────────────────────────────────────────────

class RedisPhraseCache(CachePort):
    """Redis-backed cache — persistent across restarts, TTL-controlled."""

    _KEY_PREFIX = "tts:phrase:"

    def __init__(
        self,
        redis_url: str = "redis://redis:6379/2",
        ttl_seconds: int = 86400,   # 24 hours
    ) -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._redis = None
        self._hits = 0
        self._misses = 0

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._url, decode_responses=False)
            await self._redis.ping()
            logger.info("redis_phrase_cache_connected", url=self._url)
        except Exception as exc:
            logger.warning("redis_phrase_cache_unavailable", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def get(self, key: str) -> PhraseCacheEntry | None:
        if not self._redis:
            self._misses += 1
            return None
        try:
            raw = await self._redis.get(self._KEY_PREFIX + key)
            if raw is None:
                self._misses += 1
                return None
            data = json.loads(raw.decode())
            audio_bytes = bytes.fromhex(data["audio_hex"])
            entry = PhraseCacheEntry(
                key=key,
                audio_bytes=audio_bytes,
                format=AudioFormat(data["format"]),
                sample_rate=data["sample_rate"],
                duration_s=data["duration_s"],
                char_count=data["char_count"],
                created_at=data["created_at"],
                hits=data.get("hits", 0) + 1,
            )
            self._hits += 1
            # Refresh TTL on access
            await self._redis.expire(self._KEY_PREFIX + key, self._ttl)
            return entry
        except Exception:
            logger.exception("redis_cache_get_error", key=key)
            self._misses += 1
            return None

    async def set(self, entry: PhraseCacheEntry) -> None:
        if not self._redis:
            return
        try:
            data = json.dumps({
                "audio_hex": entry.audio_bytes.hex(),
                "format": entry.format,
                "sample_rate": entry.sample_rate,
                "duration_s": entry.duration_s,
                "char_count": entry.char_count,
                "created_at": entry.created_at,
                "hits": entry.hits,
            }, ensure_ascii=False)
            await self._redis.setex(self._KEY_PREFIX + entry.key, self._ttl, data)
        except Exception:
            logger.exception("redis_cache_set_error", key=entry.key)

    async def delete(self, key: str) -> None:
        if self._redis:
            await self._redis.delete(self._KEY_PREFIX + key)

    async def clear(self) -> None:
        if not self._redis:
            return
        try:
            keys = await self._redis.keys(self._KEY_PREFIX + "*")
            if keys:
                await self._redis.delete(*keys)
            self._hits = 0
            self._misses = 0
        except Exception:
            logger.exception("redis_cache_clear_error")

    async def stats(self) -> dict:
        total = self._hits + self._misses
        n_keys = 0
        if self._redis:
            try:
                keys = await self._redis.keys(self._KEY_PREFIX + "*")
                n_keys = len(keys)
            except Exception:
                pass
        return {
            "backend": "redis",
            "entries": n_keys,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "available": self._redis is not None,
        }


# ── Tiered: L1 + L2 ──────────────────────────────────────────────────────────

class TieredPhraseCache(CachePort):
    """L1 (memory LRU) → L2 (Redis) tiered cache."""

    def __init__(self, l1: InMemoryPhraseCache, l2: RedisPhraseCache) -> None:
        self._l1 = l1
        self._l2 = l2

    async def start(self) -> None:
        await self._l1.start()
        await self._l2.start()

    async def stop(self) -> None:
        await self._l1.stop()
        await self._l2.stop()

    async def get(self, key: str) -> PhraseCacheEntry | None:
        # L1 hit
        entry = await self._l1.get(key)
        if entry is not None:
            return entry
        # L2 hit → promote to L1
        entry = await self._l2.get(key)
        if entry is not None:
            await self._l1.set(entry)
            return entry
        return None

    async def set(self, entry: PhraseCacheEntry) -> None:
        await asyncio.gather(
            self._l1.set(entry),
            self._l2.set(entry),
        )

    async def delete(self, key: str) -> None:
        await asyncio.gather(
            self._l1.delete(key),
            self._l2.delete(key),
        )

    async def clear(self) -> None:
        await asyncio.gather(
            self._l1.clear(),
            self._l2.clear(),
        )

    async def stats(self) -> dict:
        l1_stats, l2_stats = await asyncio.gather(
            self._l1.stats(),
            self._l2.stats(),
        )
        return {"l1": l1_stats, "l2": l2_stats}
