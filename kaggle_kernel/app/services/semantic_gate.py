from __future__ import annotations
from typing import List
from app.schemas.ceo import CanonicalEvidenceObject
from app.services.variable_registry import are_comparable

def filter_comparable(evs: List[CanonicalEvidenceObject], target_variable: str, target_statistic: str = None, target_window: float = None):
    """Keep only CEOs whose semantics allow comparison to target."""
    out = []
    for e in evs:
        ok, why = are_comparable(e.variable, e.statistic, e.accumulation_window_hours,
                                 target_variable, target_statistic or e.statistic, target_window)
        if ok:
            out.append(e)
        else:
            # keep warnings/advisory separately — never filter them via semantic gate
            if e.evidence_class in ("warning", "advisory"):
                out.append(e)
            else:
                e.extra["semantic_gate_reject"] = why
    return out
