"""Member-level Open-Meteo ensemble adapter.

A deterministic mean returned by an endpoint is deliberately rejected: it is not an ensemble.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from typing import Any
import httpx

from app.adapters.base import WeatherSourceAdapter
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance

URL_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"


class OpenMeteoEnsembleAdapter(WeatherSourceAdapter):
    source_name = "GEFS"
    supported_evidence_classes = ["forecast"]
    supported_variables = ["temperature_2m", "precipitation_amount"]

    async def fetch(self, lat: float, lon: float, **kwargs) -> dict[str, Any]:
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation",
                  "models": "gfs_seamless", "timezone": "UTC", "forecast_days": kwargs.get("forecast_days", 3)}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(URL_ENSEMBLE, params=params)
            response.raise_for_status()
            return response.json()

    def normalize(self, raw: dict[str, Any], lat: float, lon: float, **kwargs) -> list[CanonicalEvidenceObject]:
        hourly = raw.get("hourly", {})
        fields: list[tuple[str, int, str]] = []
        for key in hourly:
            match = re.match(r"^(temperature_2m|precipitation)_member_?(\d+)$", key)
            if match:
                fields.append((match.group(1), int(match.group(2)), key))
        if not fields:
            return []
        geometry = Geometry(type="GridCell", coordinates=[lon, lat])
        results: list[CanonicalEvidenceObject] = []
        for index, raw_time in enumerate(hourly.get("time", [])[:48]):
            try:
                valid = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                valid = valid if valid.tzinfo else valid.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            for variable, member_id, key in fields:
                values = hourly.get(key, [])
                if index >= len(values) or values[index] is None:
                    continue
                results.append(CanonicalEvidenceObject(
                    source="GEFS", evidence_class="forecast", variable=variable, value=float(values[index]),
                    unit="C" if variable == "temperature_2m" else "mm",
                    statistic="instant" if variable == "temperature_2m" else "accumulation",
                    geometry=geometry, valid_from=valid, valid_to=valid, ensemble_member=member_id,
                    accumulation_window_hours=1 if variable == "precipitation_amount" else None,
                    provenance=Provenance(original_source="Open-Meteo ensemble API", transformations=["member-level ensemble retrieval"]),
                ))
        return results

    async def health_check(self) -> dict[str, Any]:
        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(URL_ENSEMBLE, params={"latitude": 21.14, "longitude": 79.08, "hourly": "temperature_2m", "forecast_days": 1})
                response.raise_for_status()
            return {"available": True, "latency_ms": int((time.time() - started) * 1000), "reason": "endpoint reachable; member fields validated per response"}
        except Exception as exc:
            return {"available": False, "latency_ms": int((time.time() - started) * 1000), "reason": str(exc)}
