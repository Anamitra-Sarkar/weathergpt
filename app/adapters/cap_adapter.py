"""CAP Adapter — warnings, lifecycle aware."""
from __future__ import annotations
import time
from typing import List, Dict, Any
from app.adapters.base import WeatherSourceAdapter
from app.decoders.cap_decoder import decode_cap_xml

class CapAdapter(WeatherSourceAdapter):
    source_name = "CAP"
    supported_evidence_classes = ["warning"]
    supported_variables = ["heavy_rain_warning","thunderstorm_warning","cyclone_warning","heat_warning","flood_warning","marine_warning"]

    async def fetch(self, xml_bytes: bytes = None, **kwargs) -> Any:
        if xml_bytes is not None:
            return xml_bytes
        import os
        feed = os.getenv("CAP_FEED_URL")
        if not feed:
            raise RuntimeError("CAP_FEED_URL not configured")
        import httpx
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(feed)
            response.raise_for_status()
            return response.content

    def normalize(self, raw: bytes, **kwargs) -> List:
        if not raw:
            return []
        return decode_cap_xml(raw)

    async def health_check(self) -> Dict[str, Any]:
        start=time.time()
        # CAP feed requires configured URL; if not set, show unavailable but not error
        import os
        feed = os.getenv("CAP_FEED_URL")
        if not feed:
            return {"available": False, "latency_ms": 0, "reason": "CAP_FEED_URL not configured (ready)"}
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                r=await client.get(feed)
                r.raise_for_status()
                self.normalize(r.content)
            return {"available": True, "latency_ms": int((time.time()-start)*1000), "reason": "ok"}
        except Exception as e:
            return {"available": False, "latency_ms": int((time.time()-start)*1000), "reason": str(e)}
