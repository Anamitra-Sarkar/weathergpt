"""GFS GRIB2 Adapter — isolated as unavailable until eccodes present."""
from __future__ import annotations
import time
from typing import List, Dict, Any
from app.adapters.base import WeatherSourceAdapter

class Grib2Adapter(WeatherSourceAdapter):
    source_name = "GFS"
    supported_evidence_classes = ["forecast"]
    supported_variables = ["temperature_2m","precipitation_amount"]

    async def fetch(self, **kwargs) -> Any:
        raise RuntimeError("GRIB2 decoding requires eccodes+cfgrib (install requirements-full.txt) — GFS GRIB2 unavailable (isolated)")

    def normalize(self, raw: Any, **kwargs) -> List:
        raise RuntimeError("GRIB2 unavailable — use Open-Meteo forecast adapter instead")

    async def health_check(self) -> Dict[str, Any]:
        try:
            import eccodes, cfgrib, xarray
            return {"available": False, "latency_ms": 0, "reason": "GRIB2 decoder isolated — use Open-Meteo (ready)"}
        except Exception as e:
            return {"available": False, "latency_ms": 0, "reason": f"missing eccodes/cfgrib: {e} (ready, not mocked)"}
