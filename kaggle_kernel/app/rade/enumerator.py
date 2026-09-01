"""RADE — enumerate weather scenarios from ensemble."""
from __future__ import annotations
from typing import List, Dict

def enumerate_scenarios(wio) -> List[Dict]:
    """
    From WIO rain.probability and expected_range → scenarios.
    Example: p=0.7 → [{"name":"rain","p":0.7,"precip_mm":25}, {"name":"no_rain","p":0.3,"precip_mm":0}]
    """
    rain = (wio.weather.rain or {}) if hasattr(wio.weather, "rain") else {}
    p = rain.get("probability")
    if p is None:
        p = 0.5
    p = max(0, min(1, float(p)))
    precip = rain.get("value_mm", 10)
    # two-scenario minimal MDP
    return [
        {"name": "rain", "p": p, "precip_mm": precip, "wind_kmh": (wio.weather.wind or {}).get("value_kmh", 10) if wio.weather.wind else 10},
        {"name": "no_rain", "p": 1-p, "precip_mm": 0, "wind_kmh": (wio.weather.wind or {}).get("value_kmh", 10) if wio.weather.wind else 10},
    ]
