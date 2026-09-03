"""Plain result types.  No pydantic, no framework coupling.

Each carries the provenance the caller needs to build a derived evidence object:
which model produced it, at what version, and what it was derived from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FieldMapping:
    """M1: what a native field name means."""
    canonical_variable: str
    statistic: str
    accumulation_hours: Optional[float]
    vertical_level: str
    evidence_class: str
    confidence: float
    abstained: bool
    runner_up: Optional[str] = None
    margin: Optional[float] = None
    source: str = "model"          # "model" | "rule" | "abstain"
    algorithm_version: str = ""

    @property
    def is_usable(self) -> bool:
        return not self.abstained and self.canonical_variable != "other"


@dataclass
class Slot:
    kind: str                       # LOC | TIME | CROP
    text: str
    start_token: int
    end_token: int


@dataclass
class ParsedQuery:
    """M3: structured intent extracted from a natural-language question."""
    intent: str
    intent_confidence: float
    variables: list
    slots: list
    language_hint: Optional[str] = None
    algorithm_version: str = ""

    def slot_text(self, kind: str) -> Optional[str]:
        for slot in self.slots:
            if slot.kind == kind:
                return slot.text
        return None


@dataclass
class CorrectedForecast:
    """M2: a bias-corrected value with a calibrated interval."""
    variable: str
    value: float                    # the predictive median
    quantiles: dict                 # {0.05: v, ..., 0.95: v}
    interval_low: float             # conformalised 80% interval
    interval_high: float
    interval_coverage_nominal: float
    raw_ensemble_mean: float
    correction: float               # value - raw_ensemble_mean
    algorithm_version: str = ""
    parents: list = field(default_factory=list)


@dataclass
class CalibratedProbability:
    """M4: an event probability that has been verified against reliability."""
    variable: str
    threshold: float
    probability: float
    raw_ensemble_frequency: float
    method: str                     # "csgd_isotonic" | "csgd" | "raw"
    algorithm_version: str = ""
    parents: list = field(default_factory=list)


@dataclass
class RankedSource:
    """M5: one candidate source with its learned trust score."""
    source: str
    score: float
    rank: int
    value: Optional[float] = None


@dataclass
class GateResultData:
    name: str
    loaded: bool
    reason: str
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.name, "loaded": self.loaded, "reason": self.reason,
                "headline": self.metrics, **self.extra}
