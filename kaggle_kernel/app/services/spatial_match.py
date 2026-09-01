from __future__ import annotations
import math
from typing import List, Tuple
from app.schemas.ceo import CanonicalEvidenceObject

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def distance_to_query(ev: CanonicalEvidenceObject, q_lat: float, q_lon: float) -> float:
    g = ev.geometry
    if g is None or g.coordinates is None:
        return 0.0  # polygon/district — treat as covering
    try:
        if g.type == "Point":
            lon, lat = g.coordinates[0], g.coordinates[1]
            return haversine_km(q_lat, q_lon, lat, lon)
        if g.type == "GridCell":
            # coordinates = [lon, lat]
            lon, lat = g.coordinates[0], g.coordinates[1]
            return haversine_km(q_lat, q_lon, lat, lon)
    except Exception:
        return 9999
    return 0.0

def rank_by_spatial(evs: List[CanonicalEvidenceObject], q_lat: float, q_lon: float):
    return sorted(evs, key=lambda e: distance_to_query(e, q_lat, q_lon))
