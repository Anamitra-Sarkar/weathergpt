"""Engine answers -> the app's Canonical Evidence Objects (CEOs).

Only VALIDATED entries become evidence (an unvalidated entry carries no numbers, by design).  Each model keeps its own variable and
statistic so the semantic gate downstream can refuse to mix them: a probability stays a probability, a conditional rain amount is flagged
`conditional_on_wet` and is never presented as an unconditional accumulation.  `app.schemas.ceo` is imported lazily so this package works
without the FastAPI app.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

MODEL_NAME_PREFIX = "weathergpt_events"

# target -> (CanonicalVariable, statistic, what the probability / range means)
_BINARY = {
    "thunderstorm": ("thunderstorm_probability", "probability", "thunder reported in the 3 h window"),
    "fog": ("visibility", "probability", "visibility < 1 km in the 3 h window"),
    "strong_wind": ("wind_gust", "probability", "peak wind >= 25 kt in the 3 h window"),
    "rain_3h": ("precipitation_probability", "probability", "rain reported in the 3 h window"),
    "dust": ("other", "probability", "dust raising reported in the 3 h window"),
    "hot_day": ("other", "probability", "IST-day maximum >= 40 C"),
    "cold_night": ("other", "probability", "IST-day minimum <= 5 C"),
    "heatwave_imd": ("other", "probability", "IMD heat-wave day (METAR-normals approximation)"),
    "coldwave_imd": ("other", "probability", "IMD cold-wave day (METAR-normals approximation)"),
    "rain_3day_any_2p5mm": ("precipitation_probability", "probability", ">= 2.5 mm on at least one of 3 days"),
    "rain_3day_any_15mm": ("precipitation_probability", "probability", ">= 15.6 mm on at least one of 3 days"),
    "rain_7day_any_2p5mm": ("precipitation_probability", "probability", ">= 2.5 mm on at least one of 7 days"),
}
_RANGES = {
    "temperature_range": ("temperature_2m", "instant", "C"), "tmax_range": ("temperature_max", "max", "C"),
    "tmin_range": ("temperature_min", "min", "C"), "wind_range": ("wind_speed", "instant", "m/s"),
    "humidity_range": ("humidity", "instant", "%"),
}
_WINDOW_DAYS = {"rain_3day_any_2p5mm": 3, "rain_3day_any_15mm": 3, "rain_3day_total": 3, "rain_7day_any_2p5mm": 7, "rain_7day_total": 7}


def _utc(text: str) -> datetime:
    value = datetime.fromisoformat(str(text))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _valid_window(target: str, kind: str, entry: dict) -> tuple:
    """(valid_from, valid_to, lead_hours, accumulation_hours) from the entry's own time fields."""
    if "valid_time_utc" in entry:                                    # step models: a 3 h window centred on the valid time
        centre = _utc(entry["valid_time_utc"])
        return centre - timedelta(hours=1.5), centre + timedelta(hours=1.5), float(entry["lead_h"]), 3.0
    if "ist_day" in entry:                                           # IST day = 18:30 UTC the day before .. 18:30 UTC
        start = _utc(entry["ist_day"]) - timedelta(hours=5.5)
        return start, start + timedelta(hours=24), None, None
    if "utc_day" in entry:
        start = _utc(entry["utc_day"])
        return start, start + timedelta(hours=24), None, 24.0
    days = entry.get("window_days") or _WINDOW_DAYS.get(target, 1)
    start = _utc(entry["window_start_utc_day"])
    return start, start + timedelta(days=days), None, 24.0 * days


def to_ceos(result: dict, source: str = "GFS") -> list:
    """Convert one `engine.forecast` / `ForecastService.forecast` result into a list of CanonicalEvidenceObject."""
    from app.schemas.ceo import (CanonicalEvidenceObject, EvidenceClass, EvidenceSource, Geometry, GeometryType, Provenance)

    if not result.get("available"):
        return []
    lat, lon = result["location"]["lat"], result["location"]["lon"]
    init = datetime.combine(datetime.fromisoformat(result["run"]).date(), datetime.min.time(), tzinfo=timezone.utc)
    geometry = Geometry(type=GeometryType.Point, coordinates=[lon, lat])
    out = []

    def make(target, res, entry, variable, statistic, **fields):
        frm, to, lead, acc = _valid_window(target, res["kind"], entry)
        extra = {"target": target, "zone": res["zone"], "validated_in_zone": True, **fields.pop("extra", {})}
        if "lead_bucket" in entry:
            extra["lead_bucket"] = entry["lead_bucket"]
        return CanonicalEvidenceObject(
            source=EvidenceSource(source), source_type="statistical_postprocessing", evidence_class=EvidenceClass.forecast,
            variable=variable, statistic=statistic, geometry=geometry, spatial_resolution="0.25deg GFS/GEFS cell",
            model_initialization_time=init, forecast_reference_time=init, valid_from=frm, valid_to=to, forecast_lead_hours=lead,
            accumulation_window_hours=acc, model_name=f"{MODEL_NAME_PREFIX}:{target}", model_version="events-v1",
            quality_flag="validated", confidence="held-out skill demonstrated in this climate zone",
            provenance=Provenance(original_source="NOAA GFS 0.25 + GEFS mean/spread, LightGBM post-processing",
                                  original_field=target, transformations=["calibrated statistical model", "admission gate"]),
            algorithm_version="events-v1", extra=extra, **fields)

    for target, res in result["targets"].items():
        if not res.get("available"):
            continue
        for entry in (e for e in res["entries"] if e.get("validated")):
            if target in _BINARY:
                variable, statistic, meaning = _BINARY[target]
                out.append(make(target, res, entry, variable, statistic, probability=entry["probability"], extra={"event": meaning}))
            elif target in _RANGES:
                variable, statistic, unit = _RANGES[target]
                out.append(make(target, res, entry, variable, statistic, value=entry["q50"], unit=unit, extra={
                    "q10": entry["q10"], "q90": entry["q90"], "interval_lo": entry["lo"], "interval_hi": entry["hi"],
                    "nominal_coverage": entry.get("nominal_coverage"), "value_is": "median"}))
            elif target == "rain_curve":
                out.append(make(target, res, entry, "precipitation_probability", "probability", probability=entry["p_any_rain"],
                                extra={"event": "rain >= 1 mm", "p_rain_ge_2p5mm": entry["p_rainy_day"], "p_rain_ge_64p5mm": entry["p_heavy_rain"]}))
                out.append(make(target, res, entry, "rainfall_distribution", "probability", raw_value=entry["exceedance"], unit="mm",
                                extra={"imd_classes": entry["imd_classes"], "thresholds_without_skill_mm": res.get("thresholds_without_skill_mm", []),
                                       "meaning": "P(rain >= t) per threshold t in mm"}))
                amt = entry["amount_if_wet_mm"]
                if amt.get("p50") is not None:
                    out.append(make(target, res, entry, "precipitation_amount", "accumulation", value=amt["p50"], unit="mm", extra={
                        "conditional_on_wet": True, "wet_threshold_mm": 1.0, "amount_quantiles_mm": amt,
                        "amount_lower_bound": entry["amount_lower_bound"], "value_is": "median amount GIVEN rain >= 1 mm; not an unconditional forecast"}))
            elif target in ("rain_3day_total", "rain_7day_total"):
                out.append(make(target, res, entry, "rainfall_distribution", "probability", raw_value=entry["exceedance"], unit="mm",
                                extra={"thresholds_without_skill_mm": res.get("thresholds_without_skill_mm", []),
                                       "meaning": f"P({_WINDOW_DAYS[target]}-day total >= t) per threshold t in mm"}))
                amt = entry["amount_if_wet_mm"]
                if amt.get("p50") is not None:
                    out.append(make(target, res, entry, "precipitation_amount", "accumulation", value=amt["p50"], unit="mm", extra={
                        "conditional_on_wet": True, "wet_threshold_mm": entry.get("wet_mm"), "amount_quantiles_mm": amt,
                        "amount_lower_bound": entry["amount_lower_bound"], "value_is": "median total GIVEN the model's wet level; not unconditional"}))
    return out
