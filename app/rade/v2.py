"""RADE v2: deterministic, context-aware expected utility and downside-risk policy."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    name: str
    probability: float = Field(ge=0, le=1)
    precipitation_mm: float
    wind_kmh: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class DecisionAlternative(BaseModel):
    action: str
    score: float
    expected_utility: float
    downside_risk: float


class DecisionResult(BaseModel):
    recommended_action: str
    alternatives: list[DecisionAlternative] = Field(default_factory=list)
    expected_utility: float
    risk: float
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


POLICIES: dict[str, dict[str, dict[str, float]]] = {
    "spray": {
        "spray": {"dry": 20, "wet": -35, "wind_penalty": -25},
        "delay": {"dry": -5, "wet": 12}, "reschedule": {"dry": -8, "wet": 14},
    },
    "irrigate": {
        "irrigate": {"dry": 16, "wet": -14}, "delay": {"dry": -7, "wet": 12}, "partial_irrigation": {"dry": 8, "wet": 2},
    },
    "harvest": {
        "harvest_now": {"dry": 20, "wet": -30}, "delay": {"dry": -8, "wet": 10}, "partial_harvest": {"dry": 8, "wet": -2},
    },
    "marine": {
        "go": {"dry": 16, "wet": -45, "wind_penalty": -30}, "delay": {"dry": -4, "wet": 12}, "avoid": {"dry": -10, "wet": 18},
    },
    "travel": {
        "go": {"dry": 12, "wet": -22, "wind_penalty": -14}, "delay": {"dry": -5, "wet": 10}, "alternate_route": {"dry": 5, "wet": 8},
    },
}


def _domain(context: str, user_context: dict[str, Any]) -> str:
    text = f"{context} {user_context}".casefold()
    for domain, keywords in {"spray": ("spray", "pesticide"), "irrigate": ("irrigat", "sichai"), "harvest": ("harvest", "crop"), "marine": ("fish", "marine", "boat"), "travel": ("travel", "route", "drive")}.items():
        if any(keyword in text for keyword in keywords):
            return domain
    return "travel"


def generate_scenarios(wio) -> tuple[list[Scenario], list[str]]:
    rain = wio.weather.rain or {}
    evidence_ids = [item.evidence_id for item in wio.evidence if item.variable in {"precipitation_amount", "precipitation_probability", "wind_speed"}]
    members = rain.get("member_values") if isinstance(rain, dict) else None
    wind = (wio.weather.wind or {}).get("value_kmh") if wio.weather.wind else None
    if members:
        # Member provenance remains in the scenario evidence IDs; values are empirically binned.
        bins = [(0, 2, "0-2mm"), (2, 10, "2-10mm"), (10, 25, "10-25mm"), (25, 50, "25-50mm"), (50, float("inf"), ">50mm")]
        scenarios = [Scenario(name=name, probability=sum(low <= value < high for value in members) / len(members), precipitation_mm=(low if high == float("inf") else (low + high) / 2), wind_kmh=wind, evidence_ids=evidence_ids) for low, high, name in bins]
        return [scenario for scenario in scenarios if scenario.probability], ["Scenarios are derived from member-level ensemble values."]
    probability = rain.get("probability") if isinstance(rain, dict) else None
    amount = rain.get("value_mm") if isinstance(rain, dict) else None
    if probability is None or amount is None:
        return [], ["No precipitation distribution is available; RADE will not assert a weather-dependent recommendation."]
    probability = max(0.0, min(1.0, float(probability)))
    return [Scenario(name="dry", probability=1 - probability, precipitation_mm=0, wind_kmh=wind, evidence_ids=evidence_ids), Scenario(name="measurable_rain", probability=probability, precipitation_mm=float(amount), wind_kmh=wind, evidence_ids=evidence_ids)], ["Scenarios use the provider precipitation probability and amount; no ensemble distribution was available."]


def _utility(action: str, scenario: Scenario, table: dict[str, dict[str, float]]) -> float:
    values = table[action]
    wet = scenario.precipitation_mm >= 0.5
    result = values["wet" if wet else "dry"]
    if scenario.wind_kmh is not None and scenario.wind_kmh > 25:
        result += values.get("wind_penalty", 0)
    return result


def decide(wio, user_context: dict[str, Any], decision_context: str = "") -> DecisionResult:
    domain = _domain(decision_context, user_context)
    scenarios, assumptions = generate_scenarios(wio)
    evidence_ids = list(dict.fromkeys(eid for scenario in scenarios for eid in scenario.evidence_ids))
    if not scenarios:
        return DecisionResult(recommended_action="defer_decision", expected_utility=0, risk=1, confidence=0,
                              rationale="Weather evidence is insufficient for a risk-aware recommendation.", evidence_ids=evidence_ids,
                              assumptions=assumptions, rejected_actions=list(POLICIES[domain]), scenarios=[])
    risk_tolerance = str(user_context.get("risk_tolerance", "medium")).casefold()
    risk_lambda = {"low": 1.0, "medium": 0.6, "high": 0.25}.get(risk_tolerance, 0.6)
    if wio.official_warning.active and wio.official_warning.severity in {"orange", "red"}:
        risk_lambda = max(risk_lambda, 1.0)
        assumptions.append("Risk aversion increased because an official high-severity warning is active.")
    ranked: list[DecisionAlternative] = []
    for action in POLICIES[domain]:
        outcomes = [(scenario.probability, _utility(action, scenario, POLICIES[domain])) for scenario in scenarios]
        expected = sum(probability * value for probability, value in outcomes)
        downside = sum(probability * abs(value) for probability, value in outcomes if value < 0)
        ranked.append(DecisionAlternative(action=action, expected_utility=expected, downside_risk=downside, score=expected - risk_lambda * downside))
    ranked.sort(key=lambda item: item.score, reverse=True)
    best = ranked[0]
    confidence = 0.8 if wio.agreement.status in {"full_agreement", "single_source"} else 0.55
    return DecisionResult(recommended_action=best.action, alternatives=ranked[1:], expected_utility=best.expected_utility,
                          risk=best.downside_risk, confidence=float(confidence),
                          rationale=f"{best.action} has the highest risk-adjusted utility for {domain}; expected utility {best.expected_utility:.1f}, downside risk {best.downside_risk:.1f}.",
                          evidence_ids=evidence_ids, assumptions=assumptions, rejected_actions=[item.action for item in ranked[1:]], scenarios=scenarios)
