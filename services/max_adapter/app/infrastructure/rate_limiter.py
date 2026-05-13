"""Per-user sliding-window rate limiter backed by Redis (in-memory fallback).

Algorithm: fixed window (60s) — fast, low memory.
Key: max:ratelimit:{user_id}:{window}
TTL: window_seconds + 5s buffer

Defaults: 20 requests / 60 seconds per user.
Heavy users (mode: gesture processing): consume 2 tokens per request.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

from glossa_common.logging import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """Simple in-memory rate limiter — not suitable for multi-process deployments."""

    def __init__(self, limit: int = 20, window_s: int = 60) -> None:
        self._limit = limit
        self._window_s = window_s
        # {user_id: (window_start, count)}
        self._buckets: dict[int, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def check(self, user_id: int, cost: int = 1) -> tuple[bool, int]:
        """Return (allowed, remaining). Does NOT increment."""
        now = time.time()
        window_start, count = self._buckets[user_id]
        if now - window_start > self._window_s:
            window_start, count = now, 0
        remaining = max(0, self._limit - count)
        return count + cost <= self._limit, remaining

    def consume(self, user_id: int, cost: int = 1) -> tuple[bool, int]:
        """Return (allowed, remaining) and increment if allowed."""
        now = time.time()
        window_start, count = self._buckets[user_id]
        if now - window_start > self._window_s:
            window_start, count = now, 0
        if count + cost > self._limit:
            remaining = max(0, self._limit - count)
            logger.warning("rate_limit_exceeded", user_id=user_id, count=count)
            return False, remaining
        self._buckets[user_id] = (window_start, count + cost)
        return True, self._limit - count - cost


class RedisRateLimiter:
    """Redis-backed rate limiter — correct for multi-process deployments."""

    _KEY = "max:ratelimit:{user_id}:{window}"

    def __init__(
        self,
        redis_url: str = "redis://redis:6379/3",
        limit: int = 20,
        window_s: int = 60,
    ) -> None:
        self._url = redis_url
        self._limit = limit
        self._window_s = window_s
        self._redis = None
        self._fallback = InMemoryRateLimiter(limit=limit, window_s=window_s)

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._url, decode_responses=True)
            await self._redis.ping()
            logger.info("redis_rate_limiter_connected")
        except Exception as exc:
            logger.warning("redis_rate_limiter_unavailable_using_memory", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def consume(self, user_id: int, cost: int = 1) -> tuple[bool, int]:
        """Return (allowed, remaining). Atomically increment via Redis INCR."""
        if not self._redis:
            return self._fallback.consume(user_id, cost)
        try:
            window = int(time.time() // self._window_s)
            key = self._KEY.format(user_id=user_id, window=window)
            pipe = self._redis.pipeline()
            pipe.incrby(key, cost)
            pipe.expire(key, self._window_s + 5)
            results = await pipe.execute()
            new_count = results[0]
            if new_count > self._limit:
                remaining = 0
                logger.warning("rate_limit_exceeded", user_id=user_id, count=new_count)
                return False, remaining
            return True, self._limit - new_count
        except Exception:
            logger.exception("redis_rate_limiter_error", user_id=user_id)
            return self._fallback.consume(user_id, cost)
