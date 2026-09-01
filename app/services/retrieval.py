"""Concurrent, isolated source retrieval with explicit partial-result reporting."""
from __future__ import annotations

import asyncio
from datetime import date
import hashlib
import json
from typing import Any

from app.adapters.registry import REGISTRY
from app.config import settings
from app.orchestrator.retrieval_planner import RetrievalPlan
from app.schemas.ceo import CanonicalEvidenceObject
from app.services.cache import weather_cache


async def _one(source: str, lat: float, lon: float, plan: RetrievalPlan, valid_from, valid_to) -> tuple[str, list[CanonicalEvidenceObject], str | None, bool]:
    adapter = REGISTRY.get(source)
    if adapter is None:
        return source, [], "source is not configured", False
    kwargs: dict[str, Any] = {"lat": lat, "lon": lon}
    if source in {"OPEN_METEO", "GEFS"}:
        kwargs["forecast_days"] = min(16, max(1, (valid_to.date() - date.today()).days + 2))
    elif source == "ERA5":
        kwargs.update(start_date=valid_from.date().isoformat(), end_date=valid_to.date().isoformat())
    elif source == "NASA_POWER":
        kwargs.update(start=valid_from.strftime("%Y%m%d"), end=valid_to.strftime("%Y%m%d"))
    key_payload = {"source": source, "lat": round(lat, 4), "lon": round(lon, 4), "kwargs": kwargs}
    cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
    cached = await weather_cache.get(cache_key)
    if cached:
        items = [item.model_copy(deep=True) for item in cached.value]
        for item in items:
            item.extra["cache"] = {"cached": True, "retrieved_at": cached.retrieved_at.isoformat(), "expires_at": cached.expires_at.isoformat(), "stale": False}
        return source, items, None, True
    try:
        raw = await asyncio.wait_for(adapter.fetch(**kwargs), timeout=settings.source_timeout_seconds)
        items = adapter.normalize(raw, **kwargs)
        for item in items:
            item.retrieval_timestamp = item.retrieval_timestamp or __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            item.extra["cache"] = {"cached": False}
        await weather_cache.put(cache_key, items, settings.forecast_cache_ttl_seconds)
        return source, items, None, False
    except Exception as exc:
        return source, [], str(exc), False


async def retrieve(plan: RetrievalPlan, *, lat: float, lon: float, valid_from, valid_to) -> tuple[list[CanonicalEvidenceObject], dict[str, Any]]:
    results = await asyncio.gather(*[_one(source, lat, lon, plan, valid_from, valid_to) for source in plan.sources])
    evidence: list[CanonicalEvidenceObject] = []
    status: dict[str, Any] = {"sources": {}, "partial": False}
    for source, items, error, cached in results:
        evidence.extend(items)
        status["sources"][source] = {"status": "ok" if not error else "unavailable", "count": len(items), "error": error, "cached": cached}
        status["partial"] |= error is not None
    return evidence, status
