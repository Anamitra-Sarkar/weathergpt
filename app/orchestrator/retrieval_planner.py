"""Deterministic retrieval planning; language models never select data sources."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


EvidenceClassName = Literal["forecast", "warning", "observation", "reanalysis", "radar", "satellite"]


class RetrievalPlan(BaseModel):
    variables: list[str]
    evidence_classes: list[EvidenceClassName]
    sources: list[str]
    need_history: bool = False
    need_warnings: bool = False
    need_ensemble: bool = False
    decision_context: str | None = None
    reasons: list[str] = Field(default_factory=list)


def _contains(text: str, *words: str) -> bool:
    return any(word in text for word in words)


def build_retrieval_plan(question: str, horizon: str, decision_type: str | None = None) -> RetrievalPlan:
    text = question.casefold()
    decision = decision_type
    if not decision:
        if _contains(text, "spray", "pesticide", "chhidak", "छिड़क"):
            decision = "spray"
        elif _contains(text, "irrigat", "water crop", "sichai", "सिंचाई"):
            decision = "irrigate"
        elif _contains(text, "harvest", "cut crop", "katai", "कटाई"):
            decision = "harvest"
        elif _contains(text, "fish", "fishing", "marine", "boat", "sea"):
            decision = "marine"
        elif _contains(text, "travel", "go", "drive", "route"):
            decision = "travel"

    variables: list[str] = []
    if _contains(text, "rain", "baarish", "barish", "बरसात", "precip") or decision:
        variables.extend(["precipitation_amount", "precipitation_probability"])
    if _contains(text, "temperature", "temp", "hot", "cold", "mausam", "मौसम"):
        variables.append("temperature_2m")
    if _contains(text, "wind", "gust", "hawa", "हवा") or decision in {"spray", "marine", "travel"}:
        variables.extend(["wind_speed", "wind_gust"])
    if not variables:
        variables = ["temperature_2m", "precipitation_amount"]
    variables = list(dict.fromkeys(variables))

    classes: list[EvidenceClassName] = ["forecast"]
    sources = ["OPEN_METEO"]
    reasons = ["forecast requested"]
    need_warnings = _contains(text, "warning", "alert", "heavy rain", "cyclone") or decision is not None
    if need_warnings:
        classes.append("warning")
        sources.extend(["CAP", "IMD"])
        reasons.append("official warnings relevant")
    need_ensemble = decision is not None or _contains(text, "probability", "chance", "uncertain")
    if need_ensemble:
        sources.append("GEFS")
        reasons.append("uncertainty needed for decision/probability")
    need_history = horizon == "climate" or _contains(text, "usual", "history", "climate", "normal")
    if need_history:
        classes.append("reanalysis")
        sources.extend(["ERA5", "NASA_POWER"])
        reasons.append("historical context requested")
    return RetrievalPlan(
        variables=variables, evidence_classes=classes, sources=list(dict.fromkeys(sources)),
        need_history=need_history, need_warnings=need_warnings, need_ensemble=need_ensemble,
        decision_context=decision, reasons=reasons,
    )


# Compatibility for the first public prototype.  New code consumes RetrievalPlan.
def plan_retrieval(horizon: str, intent: str = "") -> list[str]:
    return build_retrieval_plan(intent, horizon).evidence_classes


def sources_for_classes(classes: list[str]) -> list[str]:
    source_map = {
        "forecast": ["OPEN_METEO"], "warning": ["CAP", "IMD"],
        "reanalysis": ["ERA5", "NASA_POWER"], "observation": ["NASA_POWER"],
        "radar": [], "satellite": [],
    }
    return list(dict.fromkeys(source for cls in classes for source in source_map.get(cls, [])))


def should_use_radar(horizon: str) -> bool:
    return False  # no configured radar adapter; planner must not claim one exists


def should_use_ensemble(horizon: str) -> bool:
    return horizon in {"short", "medium"}
