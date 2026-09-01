"""Small TTL cache that always retains freshness metadata with cached values."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio


@dataclass
class CacheEntry:
    value: Any
    retrieved_at: datetime
    expires_at: datetime

    @property
    def stale(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class TTLCache:
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str, allow_stale: bool = False) -> CacheEntry | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry and (allow_stale or not entry.stale):
                self.hits += 1
                return entry
            self.misses += 1
            return None

    async def put(self, key: str, value: Any, ttl_seconds: int) -> CacheEntry:
        now = datetime.now(timezone.utc)
        entry = CacheEntry(value=value, retrieved_at=now, expires_at=now + timedelta(seconds=ttl_seconds))
        async with self._lock:
            self._entries[key] = entry
        return entry

    def status(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"available": True, "entries": len(self._entries), "hit_rate": self.hits / total if total else 0.0}


weather_cache = TTLCache()
