"""Token-bucket FPS throttler for incoming keyframe streams.

One token is issued every (1 / max_fps) seconds.  The bucket holds at
most `burst` tokens so brief stalls don't get punished harshly.
When a frame arrives with no token available the caller should:
  1. Drop the frame, and
  2. Send a BACKPRESSURE signal to the client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ThrottleResult:
    allowed: bool
    tokens_remaining: float
    suggested_fps: float        # hint to send back in BACKPRESSURE


class TokenBucketThrottle:
    """Thread-safe (single async task) token-bucket rate limiter."""

    def __init__(self, max_fps: float = 30.0, burst: float = 5.0) -> None:
        if max_fps <= 0:
            raise ValueError("max_fps must be > 0")
        self._max_fps = max_fps
        self._burst = burst
        self._tokens: float = burst
        self._last_refill: float = time.monotonic()

    def check(self) -> ThrottleResult:
        """Non-blocking check — call once per incoming frame."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return ThrottleResult(allowed=True, tokens_remaining=self._tokens, suggested_fps=self._max_fps)

        # No token → calculate a safe suggestion (current actual rate)
        suggested = max(1.0, self._max_fps * 0.5)
        return ThrottleResult(allowed=False, tokens_remaining=0.0, suggested_fps=suggested)

    def update_max_fps(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._max_fps = fps
        # Don't carry over tokens from a higher rate — reset to safe level
        self._tokens = min(self._tokens, self._burst)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._max_fps
        self._tokens = min(self._burst, self._tokens + new_tokens)
        self._last_refill = now

    @property
    def max_fps(self) -> float:
        return self._max_fps
