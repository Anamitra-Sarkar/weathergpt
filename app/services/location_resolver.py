from __future__ import annotations
from typing import Optional, Dict
import re
from app.schemas.location import ResolvedLocation, GAZETTEER

class LocationNotFoundError(Exception):
    def __init__(self, raw: str, message: str = "Location not found"):
        self.raw = raw
        super().__init__(f"{message}: {raw}")

class LocationAmbiguousError(Exception):
    def __init__(self, raw: str, candidates: list, message: str = "Location ambiguous"):
        self.raw = raw
        self.candidates = candidates
        super().__init__(f"{message}: {raw} -> {candidates}")

# Simple cache
_cache: Dict[str, ResolvedLocation] = {}

# Minimal pincode -> district mapping for demo (real would query POST API)
PINCODE_MAP = {
    "440001": {"lat": 21.1458, "lon": 79.0882, "district": "Nagpur", "state": "Maharashtra"},
    "400001": {"lat": 18.9388, "lon": 72.8347, "district": "Mumbai", "state": "Maharashtra"},
    "110001": {"lat": 28.6139, "lon": 77.2090, "district": "New Delhi", "state": "Delhi"},
    "700001": {"lat": 22.5726, "lon": 88.3639, "district": "Kolkata", "state": "West Bengal"},
    "600001": {"lat": 13.0827, "lon": 80.2707, "district": "Chennai", "state": "Tamil Nadu"},
    "560001": {"lat": 12.9716, "lon": 77.5946, "district": "Bengaluru Urban", "state": "Karnataka"},
    "411001": {"lat": 18.5204, "lon": 73.8567, "district": "Pune", "state": "Maharashtra"},
}

def resolve_location(raw: str) -> ResolvedLocation:
    if not raw or not raw.strip():
        raise LocationNotFoundError(raw, "Location required but empty")
    key = raw.strip().lower()
    if key in _cache:
        return _cache[key]
    # pincode like 440001
    if re.match(r"^\d{6}$", key):
        if key in PINCODE_MAP:
            v = PINCODE_MAP[key]
            loc = ResolvedLocation(raw=raw, lat=v["lat"], lon=v["lon"], district=v["district"], state=v["state"], pincode=key, confidence=0.95, source="pincode", normalized_name=v["district"], administrative_hierarchy=[v["district"], v["state"]], resolution_method="pincode")
            _cache[key] = loc
            return loc
        else:
            raise LocationNotFoundError(raw, f"Unknown pincode {key} — not in PINCODE_MAP, cannot fabricate")
    # lat,lon like "21.14,79.09"
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", raw)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise LocationNotFoundError(raw, "Invalid coordinates")
        loc = ResolvedLocation(raw=raw, lat=lat, lon=lon, confidence=1.0, source="gps", normalized_name=f"{lat:.5f},{lon:.5f}", resolution_method="coordinates")
        _cache[key] = loc
        return loc
    # "near X" handling
    if key.startswith("near "):
        inner = key[5:].strip()
        return resolve_location(inner)
    # gazetteer lookup — collect all matches
    matches = []
    for k, v in GAZETTEER.items():
        if k in key:
            matches.append((k, v))
    if len(matches) == 1:
        k, v = matches[0]
        loc = ResolvedLocation(raw=raw, lat=v["lat"], lon=v["lon"], district=v["district"], state=v["state"], confidence=0.9, source="gazetteer", normalized_name=k.title(), administrative_hierarchy=[v["district"], v["state"]], resolution_method="offline_gazetteer")
        _cache[key] = loc
        return loc
    if len(matches) > 1:
        raise LocationAmbiguousError(raw, [m[0] for m in matches])
    # No match — do NOT fallback to Nagpur. Raise not found.
    raise LocationNotFoundError(raw, "Location not in gazetteer and no pincode/gps — cannot fabricate coordinates")

def is_low_confidence(loc: ResolvedLocation) -> bool:
    return loc.confidence < 0.5


def extract_location(text: str) -> ResolvedLocation | None:
    """Extract only unambiguous, locally resolvable locations from a user utterance."""
    if not text:
        return None
    coordinate = re.search(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)", text)
    if coordinate:
        return resolve_location(coordinate.group(0))
    pincode = re.search(r"\b\d{6}\b", text)
    if pincode:
        return resolve_location(pincode.group(0))
    hits = [name for name in GAZETTEER if re.search(rf"\b{re.escape(name)}\b", text, re.I)]
    if len(hits) == 1:
        return resolve_location(hits[0])
    if len(hits) > 1:
        raise LocationAmbiguousError(text, hits)
    return None
