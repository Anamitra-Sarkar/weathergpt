"""
Variable registry — semantic normalization gate.
Maps native field names → canonical variable + statistic + allowed accumulation windows.
If semantics differ, values are NOT comparable (never averaged).
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

# Canonical registry — extensible via variable_registry.yaml
DEFAULT_REGISTRY: Dict[str, Dict] = {
    # precipitation family — NOT interchangeable
    "apcp": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "kg m-2", "note": "GRIB APCP, check accumulation_window"},
    "tp": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm"},
    "precipitation": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm"},
    "rainfall": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm"},
    "rain": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm"},
    "precipitation_probability": {"canonical": "precipitation_probability", "statistic": "probability", "unit": "%"},
    "pop": {"canonical": "precipitation_probability", "statistic": "probability", "unit": "%"},
    "rain_rate": {"canonical": "precipitation_rate", "statistic": "instant", "unit": "mm/h"},
    "prate": {"canonical": "precipitation_rate", "statistic": "instant", "unit": "mm/h"},
    # temperature
    "t2m": {"canonical": "temperature_2m", "statistic": "instant", "unit": "K"},
    "2t": {"canonical": "temperature_2m", "statistic": "instant", "unit": "K"},
    "temperature": {"canonical": "temperature_2m", "statistic": "instant", "unit": "C"},
    "tmax": {"canonical": "temperature_max", "statistic": "max", "unit": "C"},
    "tmin": {"canonical": "temperature_min", "statistic": "min", "unit": "C"},
    # wind
    "wind_speed": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s"},
    "wind_gust": {"canonical": "wind_gust", "statistic": "instant", "unit": "m/s"},
    "u10": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s"},
    "v10": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s"},
    # warnings
    "heavy rainfall": {"canonical": "heavy_rain_warning", "statistic": "categorical", "unit": None},
    "heavy_rain_warning": {"canonical": "heavy_rain_warning", "statistic": "categorical", "unit": None},
}

def load_registry(path: Optional[str] = None) -> Dict[str, Dict]:
    p = Path(path) if path else Path(__file__).parent / "variable_registry.yaml"
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {}
            return {k.lower(): v for k, v in data.items()}
    return DEFAULT_REGISTRY

REGISTRY = load_registry()

def normalize_field(raw_field: str) -> Optional[Dict]:
    """Return canonical entry or None if unknown."""
    if not raw_field:
        return None
    key = raw_field.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    # fuzzy: strip spaces/underscores
    key2 = key.replace(" ", "_").replace("-", "_")
    if key2 in REGISTRY:
        return REGISTRY[key2]
    return None

def are_comparable(var_a: str, stat_a: str, window_a: Optional[float],
                   var_b: str, stat_b: str, window_b: Optional[float]) -> Tuple[bool, str]:
    """Semantic gate — only comparable if canonical variable + statistic + compatible window."""
    if var_a != var_b:
        return False, f"different variables {var_a} vs {var_b}"
    if stat_a != stat_b:
        return False, f"different statistic {stat_a} vs {stat_b}"
    if stat_a == "accumulation" and stat_b == "accumulation":
        if window_a is not None and window_b is not None and window_a != window_b:
            return False, f"accumulation window mismatch {window_a}h vs {window_b}h"
    return True, "comparable"

# --- ML helper: gold pairs for training semantic classifier ---
def gold_pairs():
    pairs = []
    for raw, entry in DEFAULT_REGISTRY.items():
        pairs.append((raw, entry["canonical"], entry["statistic"]))
    return pairs
