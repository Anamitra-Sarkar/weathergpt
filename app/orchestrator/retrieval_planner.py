"""Retrieval planner — decides which evidence classes to fetch for the horizon."""
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime

# horizon → required evidence classes (from your 16-step plan)
HORIZON_PLAN = {
    "nowcast": ["observation", "nowcast", "warning", "radar", "satellite"],
    "short":   ["observation", "forecast", "warning", "radar"],  # 0-3 days: IMD + GFS + warnings
    "medium":  ["forecast", "warning", "climate"],  # 3-10 days: GFS/GEFS + warnings
    "climate": ["climate", "advisory"],  # >10 days: ERA5/history only
}

def plan_retrieval(horizon: str, intent: str = "") -> List[str]:
    classes = HORIZON_PLAN.get(horizon, HORIZON_PLAN["short"])
    # domain hints
    if "pesticide" in intent.lower() or "spray" in intent.lower():
        # agriculture advisory is useful for pesticide decision
        if "advisory" not in classes:
            classes = classes + ["advisory"]
    if "marine" in intent.lower() or "fish" in intent.lower():
        if "advisory" not in classes:
            classes = classes + ["advisory"]
    return classes

def sources_for_classes(classes: List[str]) -> List[str]:
    mapping = {
        "observation": ["IMD"],
        "forecast": ["IMD", "OPEN_METEO", "GFS"],
        "nowcast": ["IMD"],
        "warning": ["IMD", "CAP"],
        "radar": ["RADAR"],
        "satellite": ["INSAT"],
        "climate": ["ERA5"],
        "advisory": ["OTHER"],
    }
    srcs = []
    for c in classes:
        srcs.extend(mapping.get(c, []))
    # dedup preserve order
    seen = set()
    out = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def should_use_radar(horizon: str) -> bool:
    return horizon in ("nowcast", "short")

def should_use_ensemble(horizon: str) -> bool:
    return horizon in ("short", "medium")
