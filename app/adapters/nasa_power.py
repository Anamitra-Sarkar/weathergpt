"""NASA POWER Adapter — historical independent observations."""
from __future__ import annotations
import time
from typing import List, Dict, Any
import httpx
from app.adapters.base import WeatherSourceAdapter
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance
from datetime import datetime, timezone

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

class NasaPowerAdapter(WeatherSourceAdapter):
    source_name = "NASA_POWER"
    supported_evidence_classes = ["reanalysis","observation"]
    supported_variables = ["temperature_2m","precipitation_amount"]

    async def fetch(self, lat: float, lon: float, start: str = "20240101", end: str = "20240102", **kwargs) -> Dict[str, Any]:
        params = {"parameters": "T2M,PRECTOTCORR", "community": "AG", "longitude": lon, "latitude": lat, "start": start, "end": end, "format": "JSON"}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(POWER_URL, params=params)
            r.raise_for_status()
            return r.json()

    def normalize(self, raw: Dict[str, Any], lat: float, lon: float, **kwargs) -> List[CanonicalEvidenceObject]:
        props = raw.get("properties", {}).get("parameter", {})
        t2m = props.get("T2M", {})
        precip = props.get("PRECTOTCORR", {})
        out: List[CanonicalEvidenceObject] = []
        geom = Geometry(type="Point", coordinates=[lon, lat])
        for k, v in t2m.items():
            try:
                # POWER hourly keys like 2024010100
                dt = datetime.strptime(k, "%Y%m%d%H")
                dt = dt.replace(tzinfo=timezone.utc)
                out.append(CanonicalEvidenceObject(source="NASA_POWER", evidence_class="observation", variable="temperature_2m", value=float(v), unit="C", statistic="instant", geometry=geom, observed_at=dt, valid_from=dt, valid_to=dt, provenance=Provenance(original_source="NASA POWER", original_unit="C", transformations=["fetched POWER"])))
            except: continue
        for k, v in precip.items():
            try:
                dt = datetime.strptime(k, "%Y%m%d%H")
                dt = dt.replace(tzinfo=timezone.utc)
                out.append(CanonicalEvidenceObject(source="NASA_POWER", evidence_class="observation", variable="precipitation_amount", value=float(v), unit="mm", statistic="accumulation", geometry=geom, valid_from=dt, valid_to=dt, accumulation_window_hours=1, provenance=Provenance(original_source="NASA POWER", original_unit="mm", transformations=["fetched POWER"])))
            except: continue
        return out

    async def health_check(self) -> Dict[str, Any]:
        start=time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r=await client.get(POWER_URL, params={"parameters":"T2M","community":"AG","longitude":79.08,"latitude":21.14,"start":"20240101","end":"20240101","format":"JSON"})
                r.raise_for_status()
            return {"available": True, "latency_ms": int((time.time()-start)*1000), "reason": "ok"}
        except Exception as e:
            return {"available": False, "latency_ms": int((time.time()-start)*1000), "reason": str(e)}
