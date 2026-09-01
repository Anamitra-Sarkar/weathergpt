from __future__ import annotations
from typing import Optional, Dict
import re
from app.schemas.location import ResolvedLocation, GAZETTEER

def resolve_location(raw: str) -> ResolvedLocation:
    if not raw:
        raw = "nagpur"
    key = raw.strip().lower()
    # pincode like 440001
    if re.match(r"^\d{6}$", key):
        # map pincode prefix to nearest city for demo
        return ResolvedLocation(raw=raw, lat=21.1458, lon=79.0882, district="Nagpur", state="Maharashtra", pincode=raw, confidence=0.6)
    # lat,lon like "21.14,79.09"
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", raw)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return ResolvedLocation(raw=raw, lat=lat, lon=lon, confidence=1.0, source="gps")
    # gazetteer lookup
    for k, v in GAZETTEER.items():
        if k in key:
            return ResolvedLocation(raw=raw, lat=v["lat"], lon=v["lon"], district=v["district"], state=v["state"], confidence=0.9)
    # fallback — geocode would call Nominatim here; we return Nagpur with low confidence
    return ResolvedLocation(raw=raw, lat=21.1458, lon=79.0882, district="Nagpur", state="Maharashtra", confidence=0.4, source="fallback")

def is_low_confidence(loc: ResolvedLocation) -> bool:
    return loc.confidence < 0.5
