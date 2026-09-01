from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel

class ResolvedLocation(BaseModel):
    raw: str
    lat: float
    lon: float
    district: Optional[str] = None
    state: Optional[str] = None
    block: Optional[str] = None
    pincode: Optional[str] = None
    confidence: float = 0.8
    source: str = "nominatim_cache"
    candidates: List[dict] = []

# Minimal in-memory gazetteer for demo/offline
GAZETTEER = {
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "district": "Nagpur", "state": "Maharashtra"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "district": "Mumbai", "state": "Maharashtra"},
    "delhi": {"lat": 28.6139, "lon": 77.2090, "district": "New Delhi", "state": "Delhi"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "district": "Kolkata", "state": "West Bengal"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "district": "Chennai", "state": "Tamil Nadu"},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "district": "Bengaluru Urban", "state": "Karnataka"},
    "pune": {"lat": 18.5204, "lon": 73.8567, "district": "Pune", "state": "Maharashtra"},
    "malegaon": {"lat": 20.5579, "lon": 74.5287, "district": "Nashik", "state": "Maharashtra"},
}
