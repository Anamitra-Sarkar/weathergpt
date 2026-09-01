from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Tuple
from app.schemas.ceo import CanonicalEvidenceObject
from app.services.temporal_align import staleness_hours
from app.services.spatial_match import distance_to_query

AUTHORITY = {
    "CAP": 1.0,
    "IMD": 0.95,
    "RADAR": 0.85,
    "INSAT": 0.8,
    "GFS": 0.7,
    "WRF": 0.75,
    "ERA5": 0.5,
    "OPEN_METEO": 0.7,
    "GEFS": 0.72,
    "NASA_POWER": 0.65,
    "OTHER": 0.5,
}

def score_evidence(ev: CanonicalEvidenceObject, q_lat: float, q_lon: float, now: datetime) -> float:
    auth = AUTHORITY.get(ev.source, 0.5)
    stale = min(staleness_hours(ev, now), 72) / 72  # 0 fresh, 1 stale
    freshness = 1 - stale
    dist = distance_to_query(ev, q_lat, q_lon)
    spatial = 1 / (1 + dist/50)  # 50km half-decay
    quality = 1.0
    if ev.quality_flag and "bad" in ev.quality_flag.lower():
        quality = 0.2
    # warnings get authority boost
    if ev.evidence_class == "warning":
        auth = min(1.0, auth + 0.1)
    return 0.4*auth + 0.25*freshness + 0.20*spatial + 0.15*quality

def rank(evs: List[CanonicalEvidenceObject], q_lat: float, q_lon: float, now: datetime = None):
    now = now or datetime.now(timezone.utc)
    scored: List[Tuple[float, CanonicalEvidenceObject]] = [(score_evidence(e, q_lat, q_lon, now), e) for e in evs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def detect_disagreements(scored, threshold_mm: float = 10.0):
    """Simple: if precipitation_amount values spread > threshold, flag disagreement."""
    vals = [e.value for _, e in scored if e.variable == "precipitation_amount" and e.value is not None]
    if len(vals) < 2:
        return []
    spread = max(vals) - min(vals)
    if spread > threshold_mm:
        return [f"precipitation_amount spread {min(vals):.1f}–{max(vals):.1f} mm across sources (threshold {threshold_mm} mm)"]
    return []
