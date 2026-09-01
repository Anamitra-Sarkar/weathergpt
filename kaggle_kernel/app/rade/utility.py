"""Utility model — expected utility per action × scenario."""
from __future__ import annotations
from typing import List, Dict

# Example: pesticide spraying decision
# Spraying before rain wastes pesticide; not spraying before pest window loses yield.
# Numbers are illustrative — replace with crop-economics per your brief.

def utility(action: str, scenario: Dict) -> float:
    name = scenario["name"]
    precip = scenario["precip_mm"]
    wind = scenario["wind_kmh"]
    if action == "spray":
        if name == "rain":
            return -30  # wasted + wash-off
        if wind > 25:
            return -15  # drift risk
        return 20
    if action == "wait":
        if name == "rain":
            return 10
        # missed window small penalty
        return -5
    if action == "irrigate":
        if name == "rain" and precip > 5:
            return -10  # over-water
        if name == "no_rain":
            return 15
        return 0
    return 0

def expected_utility(action: str, scenarios: List[Dict]) -> float:
    return sum(s["p"] * utility(action, s) for s in scenarios)

def all_actions() -> List[str]:
    return ["spray", "wait", "irrigate", "harvest_now", "cover_crop"]

ACTIONS = all_actions()
