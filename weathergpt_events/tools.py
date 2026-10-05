"""Tool descriptors for the orchestrator: what each specialist model answers, where and how far it is validated.

Built from the loaded registry, not typed by hand -- the catalogue the planner LLM sees can only advertise models that
passed the admission gate, with the zones and horizon their own held-out metrics support.
"""
from __future__ import annotations

from weathergpt_events.registry import EventRegistry, zone_lead_skill

SEMANTICS = {
    "thunderstorm": "Chance of thunder at/near the place (METAR TS or VCTS) in the 3 hours around each forecast step.",
    "fog": "Chance visibility drops below 1 km in the 3 hours around each step (airport-observed definition).",
    "strong_wind": "Chance of a peak gust/mean wind of at least 25 kt (46 km/h) in the 3 hours around each step.",
    "rain_3h": "Chance of rain/drizzle/showers at the place in the 3 hours around each step.",
    "dust": "Chance of dust raising reported in the 3 hours around each step (rare; often unvalidated).",
    "temperature_range": "Hourly 2 m temperature as a 10th-50th-90th percentile range (deg C).",
    "wind_range": "Hourly mean wind speed as a percentile range (m/s).",
    "humidity_range": "Hourly relative humidity as a percentile range (%).",
    "hot_day": "Chance the IST-day maximum reaches 40 C.",
    "cold_night": "Chance the IST-day minimum drops to 5 C or below.",
    "heatwave_imd": "Chance of an IMD heat wave day (plains Tmax>=40 C and +4.5 C over normal, or >=45 C; hills >=30 C and +4.5 C).",
    "coldwave_imd": "Chance of an IMD cold wave day (plains Tmin<=10 C and -4.5 C under normal, or <=4 C).",
    "tmax_range": "Tomorrow's (or later) daily maximum temperature as a percentile range (deg C).",
    "tmin_range": "Daily minimum temperature as a percentile range (deg C).",
    "rain_curve": "Rain for the UTC day: P(rain >= t) for 16 thresholds, plus P(any rain), P(rainy day), IMD class probabilities, "
                  "and the amount IF it rains (10/50/90th percentile).  Daily window = 05:30-05:30 IST.",
    "rain_3day_any_2p5mm": "Chance of at least 2.5 mm on one or more of the next 3 UTC days.",
    "rain_3day_any_15mm": "Chance of at least 15.6 mm on one or more of the next 3 UTC days.",
    "rain_3day_total": "3-day rainfall total as an exceedance curve (and amount if it rains).",
    "rain_7day_any_2p5mm": "Chance of at least 2.5 mm on one or more of the next 7 UTC days.",
    "rain_7day_total": "7-day rainfall total as an exceedance curve ('how much rain this week').",
}
# how many forecast days ahead an answer can reach (windows start early enough that they END by day 9)
HORIZON_DAYS = {"step": 10, "day": 9, "rain": 10, "rain3": 10, "rain7": 10}


def catalogue(registry: EventRegistry) -> list:
    zones = ["himalaya_north", "northeast", "northwest_arid", "indo_gangetic", "west_coast", "east_coast",
             "south_interior", "central", "islands"]
    rows = []
    for name, gate in registry.gates.items():
        metrics = registry._metrics[name]
        validated = [z for z in zones if zone_lead_skill(metrics, z, None)[0]] if gate.passed else []
        rows.append({
            "name": name, "available": gate.passed, "why_not": None if gate.passed else gate.reason,
            "answers": SEMANTICS.get(name, metrics.get("notes", "")), "kind": metrics["kind"],
            "horizon_days": HORIZON_DAYS.get(metrics.get("table", ""), None),
            "validated_zones": validated,
            "parameters": {"type": "object", "required": ["lat", "lon"], "properties": {
                "lat": {"type": "number", "description": "latitude, degrees N (land only, 6-38 N)"},
                "lon": {"type": "number", "description": "longitude, degrees E (67-98 E)"},
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 10}}},
            "headline_held_out_skill": gate.headline})
    return rows
