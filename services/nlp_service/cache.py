from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from config import CACHE_SIZE


class LRUCache:
    """Thread-safe LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int = CACHE_SIZE) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)
