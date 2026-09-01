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
# 16+ canonical vars with compatible statistics, units, windows, evidence-class restrictions
DEFAULT_REGISTRY: Dict[str, Dict] = {
    # precipitation family — NOT interchangeable
    "apcp": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "kg m-2", "accumulation_hours": [1,3,6,24], "evidence_class": ["forecast","observation","reanalysis"], "note": "GRIB APCP, check accumulation_window"},
    "tp": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm", "accumulation_hours": [1,3,6,24], "evidence_class": ["forecast","observation"]},
    "precipitation": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm", "accumulation_hours": [1,3,6,24]},
    "rainfall": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm", "accumulation_hours": [1,3,6,24]},
    "rain": {"canonical": "precipitation_amount", "statistic": "accumulation", "unit": "mm", "accumulation_hours": [1,3,6,24]},
    "precipitation_probability": {"canonical": "precipitation_probability", "statistic": "probability", "unit": "%", "evidence_class": ["forecast"]},
    "pop": {"canonical": "precipitation_probability", "statistic": "probability", "unit": "%"},
    "precip_prob": {"canonical": "precipitation_probability", "statistic": "probability", "unit": "%"},
    "rain_rate": {"canonical": "precipitation_rate", "statistic": "instant", "unit": "mm/h", "evidence_class": ["observation","radar"]},
    "prate": {"canonical": "precipitation_rate", "statistic": "instant", "unit": "mm/h"},
    "precipitation_rate": {"canonical": "precipitation_rate", "statistic": "instant", "unit": "mm/h"},
    # temperature
    "t2m": {"canonical": "temperature_2m", "statistic": "instant", "unit": "K", "evidence_class": ["forecast","observation","reanalysis"]},
    "2t": {"canonical": "temperature_2m", "statistic": "instant", "unit": "K"},
    "temperature": {"canonical": "temperature_2m", "statistic": "instant", "unit": "C"},
    "temperature_2m": {"canonical": "temperature_2m", "statistic": "instant", "unit": "C"},
    "tmax": {"canonical": "temperature_max", "statistic": "max", "unit": "C"},
    "tmin": {"canonical": "temperature_min", "statistic": "min", "unit": "C"},
    # wind
    "wind_speed": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s", "evidence_class": ["forecast","observation"]},
    "wind_gust": {"canonical": "wind_gust", "statistic": "instant", "unit": "m/s"},
    "windgust": {"canonical": "wind_gust", "statistic": "instant", "unit": "m/s"},
    "u10": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s"},
    "v10": {"canonical": "wind_speed", "statistic": "instant", "unit": "m/s"},
    # humidity/pressure/visibility/cloud
    "humidity": {"canonical": "humidity", "statistic": "instant", "unit": "%"},
    "relative_humidity": {"canonical": "humidity", "statistic": "instant", "unit": "%"},
    "rh": {"canonical": "humidity", "statistic": "instant", "unit": "%"},
    "pressure": {"canonical": "pressure_msl", "statistic": "instant", "unit": "hPa"},
    "pressure_msl": {"canonical": "pressure_msl", "statistic": "instant", "unit": "hPa"},
    "msl": {"canonical": "pressure_msl", "statistic": "instant", "unit": "hPa"},
    "visibility": {"canonical": "visibility", "statistic": "instant", "unit": "km"},
    "cloud_cover": {"canonical": "cloud_cover", "statistic": "instant", "unit": "%"},
    "cloudcover": {"canonical": "cloud_cover", "statistic": "instant", "unit": "%"},
    # warnings — categorical, evidence_class warning only
    "heavy rainfall": {"canonical": "heavy_rain_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "heavy_rain_warning": {"canonical": "heavy_rain_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "thunderstorm": {"canonical": "thunderstorm_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "thunderstorm_warning": {"canonical": "thunderstorm_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "cyclone": {"canonical": "cyclone_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "cyclone_warning": {"canonical": "cyclone_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "heat wave": {"canonical": "heat_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "heat_warning": {"canonical": "heat_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "flood": {"canonical": "flood_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "flood_warning": {"canonical": "flood_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "marine": {"canonical": "marine_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
    "marine_warning": {"canonical": "marine_warning", "statistic": "categorical", "unit": None, "evidence_class": ["warning"]},
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


def validate_semantics(variable: str, statistic: str, unit: str | None,
                       evidence_class: str, accumulation_hours: float | None) -> tuple[bool, str]:
    """Validate a CEO against canonical semantics rather than trusting a decoder or an LLM."""
    entries = [entry for entry in DEFAULT_REGISTRY.values() if entry["canonical"] == variable]
    if not entries:
        return False, f"unknown canonical variable {variable}"
    allowed_statistics = {entry["statistic"] for entry in entries}
    if statistic not in allowed_statistics:
        return False, f"{variable} does not support statistic {statistic}"
    restrictions = [entry.get("evidence_class") for entry in entries if entry.get("evidence_class")]
    if restrictions and not any(evidence_class in allowed for allowed in restrictions):
        return False, f"{variable} is not valid for evidence class {evidence_class}"
    if statistic == "accumulation":
        if not accumulation_hours or accumulation_hours <= 0:
            return False, "accumulation requires a positive accumulation window"
        allowed_windows = {window for entry in entries for window in entry.get("accumulation_hours", [])}
        if allowed_windows and accumulation_hours not in allowed_windows:
            return False, f"unsupported accumulation window {accumulation_hours}h"
    if statistic == "probability" and unit not in {None, "%", "1", "probability"}:
        return False, f"invalid probability unit {unit}"
    return True, "valid"

# --- ML helper: gold pairs for training semantic classifier ---
def gold_pairs():
    pairs = []
    for raw, entry in DEFAULT_REGISTRY.items():
        pairs.append((raw, entry["canonical"], entry["statistic"]))
    return pairs
