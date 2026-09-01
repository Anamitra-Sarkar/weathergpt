"""Open-Meteo Forecast Adapter — real, key-free."""
from __future__ import annotations
import time
from typing import List, Dict, Any
import httpx
from app.adapters.base import WeatherSourceAdapter
from app.decoders.open_meteo import decode_open_meteo
from app.schemas.ceo import CanonicalEvidenceObject

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

class OpenMeteoForecastAdapter(WeatherSourceAdapter):
    source_name = "OPEN_METEO"
    supported_evidence_classes = ["forecast"]
    supported_variables = ["temperature_2m","precipitation_amount","precipitation_probability","wind_speed","humidity","pressure_msl","cloud_cover"]

    async def fetch(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        hourly = kwargs.get("hourly", "temperature_2m,precipitation,precipitation_probability,wind_speed_10m,relative_humidity_2m,pressure_msl,cloud_cover")
        forecast_days = kwargs.get("forecast_days", 3)
        params = {"latitude": lat, "longitude": lon, "hourly": hourly, "forecast_days": forecast_days, "timezone": "UTC"}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(FORECAST_URL, params=params)
            r.raise_for_status()
            return r.json()

    def normalize(self, raw: Dict[str, Any], lat: float, lon: float, **kwargs) -> List[CanonicalEvidenceObject]:
        return decode_open_meteo(raw, lat, lon)

    async def health_check(self) -> Dict[str, Any]:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(FORECAST_URL, params={"latitude": 21.14, "longitude": 79.08, "hourly": "temperature_2m", "forecast_days": 1})
                r.raise_for_status()
            return {"available": True, "latency_ms": int((time.time()-start)*1000), "reason": "ok"}
        except Exception as e:
            return {"available": False, "latency_ms": int((time.time()-start)*1000), "reason": str(e)}
