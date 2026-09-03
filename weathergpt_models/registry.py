"""Artifact resolution and the metrics contract gate.

A trained artifact is only allowed to serve if it can prove three things about
itself, from its own `metrics.json`:

  1. **Provenance** — which dataset it saw, that dataset's SHA-256, how the
     split was drawn, and when it was trained.  An artifact that cannot say what
     it learned from cannot be audited later.
  2. **A baseline** — at least one comparison against the thing it replaces.  A
     metric with no baseline beside it is decoration.
  3. **A margin** — its headline metric actually beats that baseline by the
     minimum recorded in `GATES` below.

Anything that fails is refused, the attribute stays `None`, and `status()`
reports the exact gate.  The caller keeps its deterministic path.

This is not defensive theatre.  This repository previously shipped a
`best.pt` fitted to `np.random.randn(500, 5)` for one epoch, next to a
`metrics.json` that declared `"dataset_kind": "real_matched_pairs"`, and nothing
in the system was able to notice.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from weathergpt_models.types import GateResultData as GateResult

REQUIRED_PROVENANCE = ("algorithm_version", "dataset_kind", "dataset_sha256",
                       "split", "trained_at")


@dataclass(frozen=True)
class Gate:
    """One model's admission test."""
    directory: str
    headline: Callable[[dict], dict]
    passes: Callable[[dict], tuple]


def _m1_headline(metrics: dict) -> dict:
    return {
        "zeroshot_macro_f1": metrics.get("test_zeroshot_macro_f1"),
        "baseline_macro_f1": (metrics.get("baselines", {})
                              .get("dict_registry_zeroshot", {}).get("macro_f1")),
        "misassignment_rate": metrics.get("test_zeroshot_misassignment_rate"),
    }


def _m1_passes(metrics: dict) -> tuple:
    model = metrics.get("test_zeroshot_macro_f1")
    baseline = (metrics.get("baselines", {})
                .get("dict_registry_zeroshot", {}).get("macro_f1"))
    misassignment = metrics.get("test_zeroshot_misassignment_rate")
    if model is None or baseline is None:
        return False, "no zero-shot macro-F1 or no dict-registry baseline recorded"
    if model <= baseline:
        return False, f"zero-shot macro-F1 {model:.3f} does not beat the dict registry {baseline:.3f}"
    if misassignment is not None and misassignment > 0.05:
        return False, (f"misassignment rate {misassignment:.3f} exceeds 0.05 — it would "
                       f"confidently map fields to the wrong variable")
    return True, f"zero-shot macro-F1 {model:.3f} vs dict registry {baseline:.3f}"


def _m2_headline(metrics: dict) -> dict:
    out = {}
    for variable, block in (metrics.get("results") or {}).items():
        test = block.get("test_spatial_holdout", {})
        out[variable] = {"crpss_vs_raw_ensemble": test.get("crpss_vs_raw_ensemble"),
                         "rmse_skill_vs_multi_model_mean": test.get("rmse_skill_vs_multi_model_mean")}
    return out


def _m2_passes(metrics: dict) -> tuple:
    results = metrics.get("results") or {}
    if not results:
        return False, "no per-variable results recorded"
    failures = []
    for variable, block in results.items():
        test = block.get("test_spatial_holdout", {})
        skill = test.get("crpss_vs_raw_ensemble")
        if skill is None:
            failures.append(f"{variable}: no CRPS skill score against the raw ensemble")
        elif skill <= 0:
            failures.append(f"{variable}: CRPSS {skill:+.3f} — no better than the raw ensemble")
    if failures:
        return False, "; ".join(failures)
    return True, "positive CRPS skill against the raw ensemble on the spatial holdout for every variable"


def _m3_headline(metrics: dict) -> dict:
    test = metrics.get("test_heldout", {})
    return {"intent_macro_f1": test.get("intent_macro_f1"),
            "slot_f1": test.get("slot_f1"),
            "baseline_intent_macro_f1": (metrics.get("baselines", {})
                                         .get("rule_based_retrieval_planner_test", {})
                                         .get("intent_macro_f1"))}


def _m3_passes(metrics: dict) -> tuple:
    test = metrics.get("test_heldout", {})
    model = test.get("intent_macro_f1")
    baseline = (metrics.get("baselines", {})
                .get("rule_based_retrieval_planner_test", {}).get("intent_macro_f1"))
    slot = test.get("slot_f1")
    if model is None or baseline is None:
        return False, "no held-out intent macro-F1 or no rule-based baseline recorded"
    if model <= baseline:
        return False, f"intent macro-F1 {model:.3f} does not beat the rule parser {baseline:.3f}"
    if slot is not None and slot < 0.5:
        return False, f"slot F1 {slot:.3f} is too low to hand spans to a geocoder"
    return True, f"intent macro-F1 {model:.3f} vs rule parser {baseline:.3f}, slot F1 {slot:.3f}"


def _m4_headline(metrics: dict) -> dict:
    out = {}
    for variable, block in (metrics.get("results") or {}).items():
        test = block.get("test_spatial_holdout", {})
        out[variable] = {"crpss_vs_raw_ensemble": test.get("crpss_vs_raw_ensemble")}
    exceedance = (metrics.get("results") or {}).get("precipitation", {}).get("exceedance", {})
    if exceedance:
        out["precipitation_1mm"] = exceedance.get("1.0", {})
    return out


def _m4_passes(metrics: dict) -> tuple:
    exceedance = (metrics.get("results") or {}).get("precipitation", {}).get("exceedance")
    if not exceedance:
        return False, "no precipitation exceedance verification recorded"
    failures = []
    for threshold, block in exceedance.items():
        calibrated = block.get("brier_csgd_isotonic")
        raw = block.get("brier_raw_ensemble_frequency")
        if calibrated is None or raw is None:
            failures.append(f">{threshold}mm: missing Brier scores")
        elif calibrated > raw:
            failures.append(f">{threshold}mm: calibrated Brier {calibrated:.5f} is worse than "
                            f"raw ensemble frequency {raw:.5f}")
    if failures:
        return False, "; ".join(failures[:3])
    return True, "calibrated Brier beats raw ensemble frequency at every threshold"


def _m5_headline(metrics: dict) -> dict:
    out = {}
    for variable, block in (metrics.get("results") or {}).items():
        test = block.get("test_spatial_holdout", {})
        out[variable] = {"ndcg@1": test.get("ndcg@1"),
                         "rmse_skill_vs_fixed_authority": test.get("rmse_skill_vs_fixed_authority")}
    return out


def _m5_passes(metrics: dict) -> tuple:
    results = metrics.get("results") or {}
    if not results:
        return False, "no per-variable results recorded"
    wins = 0
    for variable, block in results.items():
        test = block.get("test_spatial_holdout", {})
        learned = test.get("rmse_following_ranker")
        fixed = test.get("rmse_following_fixed_authority")
        if learned is not None and fixed is not None and learned < fixed:
            wins += 1
    if wins == 0:
        return False, ("following the learned ranker is no better than the fixed authority "
                       "order for any variable")
    return True, f"beats the fixed authority order on {wins}/{len(results)} variables"


GATES: dict[str, Gate] = {
    "field_mapper": Gate("m1_field_mapper_v1", _m1_headline, _m1_passes),
    "mos": Gate("m2_mos_v1", _m2_headline, _m2_passes),
    "intent": Gate("m3_intent_v1", _m3_headline, _m3_passes),
    "calibration": Gate("m4_calibration_v1", _m4_headline, _m4_passes),
    "trust_ranker": Gate("m5_trust_ranker_v1", _m5_headline, _m5_passes),
}


class ModelRegistry:
    """Loads the trained artifacts, refusing any that cannot justify itself."""

    def __init__(self, root: str | os.PathLike, *, lazy: bool = True,
                 device: str = "cpu", strict: bool = False):
        self.root = Path(root)
        self.device = device
        self.strict = strict
        self._gates: dict[str, GateResult] = {}
        self._loaded: dict[str, Any] = {}
        self._evaluate_gates()
        if not lazy:
            for name in GATES:
                getattr(self, name)

    # --- construction -------------------------------------------------------
    @classmethod
    def from_dir(cls, path: str | os.PathLike, **kwargs) -> "ModelRegistry":
        return cls(path, **kwargs)

    @classmethod
    def from_hub(cls, repo_id: str, *, revision: str = "main",
                 cache_dir: Optional[str] = None, **kwargs) -> "ModelRegistry":
        """Download the artifact bundle from the Hugging Face Hub."""
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        return cls(path, **kwargs)

    # --- gating -------------------------------------------------------------
    def _refuse(self, name: str, reason: str) -> None:
        self._gates[name] = GateResult(name, False, reason)
        if self.strict:
            raise RuntimeError(f"{name} failed its metrics gate: {reason}")

    def _evaluate_gates(self) -> None:
        for name, gate in GATES.items():
            directory = self.root / gate.directory
            metrics_path = directory / "metrics.json"
            if not metrics_path.exists():
                # A model that was never trained is not a failure to raise on;
                # strict mode is about refusing artifacts that misrepresent
                # themselves, not about demanding a full set.
                self._gates[name] = GateResult(name, False,
                                               f"no artifact at {gate.directory}/metrics.json")
                continue
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception as exc:
                self._refuse(name, f"unreadable metrics.json: {exc}")
                continue

            missing = [key for key in REQUIRED_PROVENANCE if not metrics.get(key)]
            if missing:
                self._refuse(name,
                             f"metrics.json is missing provenance {missing}; an artifact that "
                             f"cannot say what it was trained on is not admissible")
                continue

            try:
                ok, reason = gate.passes(metrics)
            except Exception as exc:
                ok, reason = False, f"gate raised {type(exc).__name__}: {exc}"

            self._gates[name] = GateResult(
                name, ok, reason, metrics=gate.headline(metrics),
                extra={"algorithm_version": metrics.get("algorithm_version"),
                       "dataset_sha256": (metrics.get("dataset_sha256") or "")[:16],
                       "split": metrics.get("split"),
                       "trained_at": metrics.get("trained_at")})
            if not ok and self.strict:
                raise RuntimeError(f"{name} failed its metrics gate: {reason}")

    def status(self) -> list:
        """What loaded, what did not, and precisely why."""
        return [self._gates[name].as_dict() for name in GATES]

    def metrics(self, name: str) -> dict:
        gate = GATES[name]
        return json.loads((self.root / gate.directory / "metrics.json").read_text())

    def _artifact_dir(self, name: str) -> Optional[Path]:
        if not self._gates[name].loaded:
            return None
        return self.root / GATES[name].directory

    def _get(self, name: str, builder):
        if name in self._loaded:
            return self._loaded[name]
        directory = self._artifact_dir(name)
        if directory is None:
            self._loaded[name] = None
            return None
        try:
            self._loaded[name] = builder(directory)
        except Exception as exc:
            self._gates[name] = GateResult(name, False, f"failed to load: {type(exc).__name__}: {exc}")
            self._loaded[name] = None
        return self._loaded[name]

    # --- models -------------------------------------------------------------
    @property
    def field_mapper(self):
        from weathergpt_models.field_mapper import FieldMapper
        return self._get("field_mapper", lambda d: FieldMapper(d, device=self.device))

    @property
    def intent(self):
        from weathergpt_models.intent import IntentParser
        return self._get("intent", lambda d: IntentParser(d, device=self.device))

    @property
    def mos(self):
        from weathergpt_models.mos import MOSCorrector
        return self._get("mos", lambda d: MOSCorrector(d, device=self.device))

    @property
    def calibration(self):
        from weathergpt_models.calibration import ProbabilityCalibrator
        return self._get("calibration", lambda d: ProbabilityCalibrator(d, device=self.device))

    @property
    def trust_ranker(self):
        from weathergpt_models.trust_ranker import TrustRanker
        return self._get("trust_ranker", lambda d: TrustRanker(d))
