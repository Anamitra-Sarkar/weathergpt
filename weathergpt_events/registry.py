"""Artifact discovery and the admission gate.

Same philosophy as `weathergpt_models.registry`: an artifact serves only if its own `metrics.json` proves
  1. provenance  - what it learned from (dataset kind + fingerprint, split, version, timestamp);
  2. a baseline  - it was measured against something it replaces (local climatology; a calibrated raw forecast);
  3. a margin    - it beats that baseline on PLACES IT NEVER SAW, in the FUTURE (`test_space`), not just in-sample.
Anything that fails is refused with the exact reason; the caller keeps its deterministic path.

Past gate, the same file drives per-(zone, lead) skill gating at request time: a model with no demonstrated skill in
the user's climate zone or at that lead is flagged `validated=False`, never silently trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from weathergpt_events.models import load_model

REQUIRED_PROVENANCE = ("algorithm_version", "dataset_kind", "dataset_sha256", "split", "trained_at")
COVERAGE_BAND = (0.70, 0.90)      # a nominal-80% range must land here on unseen places
AUC_TOLERANCE = 0.01              # ~2 standard errors of an AUC on a few thousand rows; clearly-worse still fails
MIN_BSS = 0.005                   # a model fitted to pure noise scores 0 +- ~0.003; "> 0" would admit it
MIN_AUC = 0.55                    # some real discrimination at all
MIN_MAE_GAIN = 0.01               # median must beat the raw forecast by at least 1%
MIN_N = 500                       # smallest zone/lead sample we are willing to call evidence


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    headline: dict = field(default_factory=dict)


def gate(metrics: dict) -> GateResult:
    name = metrics.get("target", "?")
    if "provenance" not in metrics:
        return GateResult(name, False, "no provenance block: cannot say what this model learned from")
    missing = [k for k in REQUIRED_PROVENANCE if not metrics["provenance"].get(k)]
    if missing:
        return GateResult(name, False, f"provenance incomplete: {missing}")
    space = metrics.get("metrics", {}).get("test_space")
    if not space:
        return GateResult(name, False, "no test_space metrics (never evaluated on unseen places)")
    kind = metrics["kind"]
    if kind in ("binary", "xent"):
        bss, auc, auc_rule = space.get("bss_vs_zone_month"), space.get("auc"), space.get("auc_rule_feature")
        head = {"bss_vs_zone_month": bss, "auc": auc, "auc_rule_feature": auc_rule}
        if bss is None:
            return GateResult(name, False, "no local-climatology baseline recorded", head)
        if bss < MIN_BSS:
            return GateResult(name, False, f"does not beat local climatology on unseen places by a meaningful margin (BSS {bss} < {MIN_BSS})", head)
        if auc is not None and auc < MIN_AUC:
            return GateResult(name, False, f"no discrimination on unseen places (AUC {auc} < {MIN_AUC})", head)
        if auc is not None and auc_rule is not None and auc < auc_rule - AUC_TOLERANCE:
            return GateResult(name, False, f"ranks worse than the raw forecast feature (AUC {auc} < {auc_rule})", head)
        return GateResult(name, True, "ok", head)
    if kind == "quantile":
        mae, raw_mae, cov = space.get("median_mae"), space.get("gfs_raw_mae"), space.get("coverage_conformal")
        head = {"median_mae": mae, "gfs_raw_mae": raw_mae, "coverage_conformal": cov}
        if mae is None or raw_mae is None:
            return GateResult(name, False, "no raw-forecast baseline recorded", head)
        if mae > raw_mae * (1 - MIN_MAE_GAIN):
            return GateResult(name, False, f"median does not beat the raw forecast by {MIN_MAE_GAIN:.0%} (MAE {mae} vs {raw_mae})", head)
        if cov is None or not COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1]:
            return GateResult(name, False, f"interval coverage {cov} outside {COVERAGE_BAND} on unseen places", head)
        return GateResult(name, True, "ok", head)
    if kind == "curve":
        detail = space.get("detail_key_thresholds") or {}
        thresholds = space.get("thresholds", {})
        key = list(detail)[:2]
        if not key:
            return GateResult(name, False, "no key-threshold detail recorded")
        head = {k: {x: thresholds.get(k, {}).get(x) for x in ("bss_vs_zone_month", "bss_vs_gfs_calibrated")} for k in key}
        if space.get("monotone_violations_calibrated", 1) != 0:
            return GateResult(name, False, "calibrated curve is not monotone", head)
        for k in key:
            b = thresholds.get(k, {}).get("bss_vs_zone_month")
            if b is None or b < MIN_BSS:
                return GateResult(name, False, f"threshold {k} mm does not beat local climatology on unseen places (BSS {b})", head)
        return GateResult(name, True, "ok", head)
    return GateResult(name, False, f"unknown kind {kind!r}")


def zone_lead_skill(metrics: dict, zone: str, lead_bucket: str | None) -> tuple:
    """-> (validated, note).  Looks the (zone, lead) up in the held-out breakdowns the trainer recorded."""
    space = metrics.get("metrics", {}).get("test_space", {})
    kind = metrics["kind"]
    if kind == "quantile":
        entry = (space.get("coverage_by_zone") or {}).get(zone)
        if not entry or entry["n"] < MIN_N:
            return False, f"no held-out evidence for zone '{zone}'"
        if not COVERAGE_BAND[0] <= entry["coverage"] <= COVERAGE_BAND[1]:
            return False, f"range under/over-covers in zone '{zone}' (coverage {entry['coverage']})"
        return True, "ok"
    if kind == "curve":
        detail = space.get("detail_key_thresholds") or {}
        verdicts = []
        for thr, d in detail.items():
            z = (d.get("by_zone") or {}).get(zone)
            verdicts.append(bool(z and z["n"] >= MIN_N and z.get("bss_vs_zone_month") is not None and z["bss_vs_zone_month"] > 0))
        if not verdicts:
            return False, "no held-out zone breakdown"
        if not any(verdicts):
            return False, f"no skill over local climatology in zone '{zone}' at any key threshold"
        return True, "ok" if all(verdicts) else f"skill in zone '{zone}' only at some thresholds"
    z = (space.get("by_zone") or {}).get(zone)
    if not z or z["n"] < MIN_N:
        return False, f"no held-out evidence for zone '{zone}'"
    if z.get("bss_vs_local_clim") is None or z["bss_vs_local_clim"] <= 0:
        return False, f"no skill over the zone's own climatology in '{zone}' (BSS {z.get('bss_vs_local_clim')})"
    if lead_bucket is not None:
        lead = (space.get("by_lead") or {}).get(lead_bucket)
        if lead and lead["n"] >= MIN_N and lead.get("bss_vs_local_clim") is not None and lead["bss_vs_local_clim"] <= 0:
            return False, f"no skill over climatology at lead '{lead_bucket}'"
    return True, "ok"


class EventRegistry:
    def __init__(self):
        self.models, self.gates, self._metrics = {}, {}, {}

    @classmethod
    def from_dir(cls, directory: str | Path) -> "EventRegistry":
        registry = cls()
        for sub in sorted(Path(directory).iterdir()):
            if not (sub / "metrics.json").exists():
                continue
            metrics = json.loads((sub / "metrics.json").read_text())
            if "skipped" in metrics or "kind" not in metrics:
                continue
            result = gate(metrics)
            registry.gates[sub.name], registry._metrics[sub.name] = result, metrics
            if result.passed:
                registry.models[sub.name] = load_model(sub)
        return registry

    def status(self) -> list:
        return [{"target": n, "loaded": g.passed, "reason": g.reason, "headline": g.headline} for n, g in self.gates.items()]

    def skill(self, target: str, zone: str, lead_bucket: str | None = None) -> tuple:
        return zone_lead_skill(self._metrics[target], zone, lead_bucket)
