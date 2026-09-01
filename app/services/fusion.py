"""Deterministic evidence fusion — source reliability + freshness + spatial + ensemble."""
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone
from app.schemas.ceo import CanonicalEvidenceObject
from app.services.ranker import score_evidence, AUTHORITY
from app.services.spatial_match import distance_to_query
from app.services.temporal_align import staleness_hours

# Source reliability by variable (simplified, will be learned from evaluation)
RELIABILITY = {
    ("IMD","temperature_2m"): 0.95,
    ("OPEN_METEO","temperature_2m"): 0.85,
    ("GFS","temperature_2m"): 0.80,
    ("ERA5","temperature_2m"): 0.90,
    ("IMD","precipitation_amount"): 0.90,
    ("OPEN_METEO","precipitation_amount"): 0.80,
    ("GFS","precipitation_amount"): 0.75,
}

def fuse(ceos: List[CanonicalEvidenceObject], q_lat: float, q_lon: float, now: datetime = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    scored = []
    for c in ceos:
        base = score_evidence(c, q_lat, q_lon, now)
        rel = RELIABILITY.get((c.source, c.variable), 0.7)
        # Adjust for lead time: longer lead → lower reliability
        lead_penalty = 1.0
        if c.forecast_lead_hours and c.forecast_lead_hours > 72:
            lead_penalty = 0.8
        # Quality flag
        quality = 1.0 if not c.quality_flag or "bad" not in c.quality_flag.lower() else 0.3
        fused_score = 0.5*base + 0.3*rel + 0.2*quality*lead_penalty
        scored.append((fused_score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Group by variable
    best_by_var: Dict[str, CanonicalEvidenceObject] = {}
    for score, c in scored:
        if c.variable not in best_by_var and c.evidence_class != "warning":
            best_by_var[c.variable] = c
    # Warning severity hierarchy separate
    warnings = [c for c in ceos if c.evidence_class=="warning"]
    # Sort warnings by severity
    severity_rank = {"green":0,"yellow":1,"orange":2,"red":3,"cancelled":-1}
    warnings_sorted = sorted(warnings, key=lambda w: severity_rank.get((w.warning_severity or "yellow").lower(),1), reverse=True)
    return {"scored": scored, "best_by_var": best_by_var, "warnings_sorted": warnings_sorted, "fusion_metadata": {"scored_count": len(scored), "reliability_used": True}}
