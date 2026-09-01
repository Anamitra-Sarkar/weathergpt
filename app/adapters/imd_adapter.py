"""IMD Adapter — ready stub, fails clearly if IMD_API_KEY missing."""
from __future__ import annotations
import os, time
from typing import List, Dict, Any
import httpx
from app.adapters.base import WeatherSourceAdapter
from app.decoders.imd_json import decode

class ImdAdapter(WeatherSourceAdapter):
    source_name = "IMD"
    supported_evidence_classes = ["forecast","observation","warning","reanalysis"]
    supported_variables = ["temperature_2m","precipitation_amount","wind_speed","heavy_rain_warning"]

    async def fetch(self, **kwargs) -> Any:
        key = os.getenv("IMD_API_KEY")
        base = os.getenv("IMD_API_BASE", "https://api.data.gov.in")
        if not key:
            raise RuntimeError("IMD_API_KEY missing — IMD source unavailable (ready, not mocked)")
        # Example: city forecast endpoint — real IMD portal requires specific resource
        # This adapter is ready but will only be used when credentials are set
        params = {"api-key": key, "format": "json", "limit": 10, **kwargs}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{base}/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69", params=params)
            r.raise_for_status()
            return r.json()

    def normalize(self, raw: Any, **kwargs) -> List:
        # raw is IMD JSON records list
        product = kwargs.get("product", "forecast")
        if isinstance(raw, dict) and "records" in raw:
            records = raw["records"]
        elif isinstance(raw, list):
            records = raw
        else:
            records = [raw]
        out=[]
        for rec in records:
            out.extend(decode(rec, product=product))
        return out

    async def health_check(self) -> Dict[str, Any]:
        if not os.getenv("IMD_API_KEY"):
            return {"available": False, "latency_ms": 0, "reason": "IMD_API_KEY not configured (ready)"}
        start=time.time()
        try:
            await self.fetch(limit=1)
            return {"available": True, "latency_ms": int((time.time()-start)*1000), "reason": "ok"}
        except Exception as e:
            return {"available": False, "latency_ms": int((time.time()-start)*1000), "reason": str(e)}
