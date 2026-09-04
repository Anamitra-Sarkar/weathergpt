# Wiring `weathergpt_models` into the serving app

**Status as of this writing: not done.** `app/main.py` and every module under
`app/services/` still run the pure deterministic path — dict lookup for field
mapping, hand-tuned weights in `ranker.py`, no bias correction, no calibrated
probabilities, no ML intent parsing, RADE fed by counted ensemble members. A
request hitting the live API today gets zero benefit from the five trained
models. This document is the exact, file-by-file map of what changes that.

The AI/ML side (`modal_jobs/`, `weathergpt_models/`) was scoped separately from
the framework (`app/`), so this integration was deliberately left to whoever
owns `app/`. Nothing below has been applied to `app/`.

---

## What you're integrating

```python
from weathergpt_models import ModelRegistry

registry = ModelRegistry.from_hub("Arko007/weathergpt-models")
# or, if you'd rather stage the bundle onto local/container disk yourself:
# ModelRegistry.from_dir("/path/to/downloaded/bundle")
```

Published 2026-09-04, private repo (2.16 GB, all 5 gate-passing models).
`from_hub` needs an HF token with read access to that repo — same
authentication `huggingface_hub` always uses (`HF_TOKEN` env var or
`huggingface-cli login`), ask for read access to be added to the repo if
you don't have it yet.

Load this **once**, at process startup, not per-request — model weights stay
resident in memory. `registry.status()` tells you what loaded and, for
anything that didn't, exactly why (a metrics-contract gate refuses an artifact
that can't prove it beats its baseline — see `weathergpt_models/registry.py`).
Each of the five attributes is `None` if its gate failed; **always check for
`None` and fall back to the existing deterministic path**, never assume a
model loaded.

```python
app = FastAPI(...)
_registry: ModelRegistry | None = None

@app.on_event("startup")
def _load_models():
    global _registry
    _registry = ModelRegistry.from_dir(settings.model_dir)  # add this setting
    logger.info("model registry: %s", _registry.status())
```

Add `weathergpt_models` as a real dependency (it's already a sibling package
in this repo — `pip install -e .` from the repo root, or add it to whatever
`requirements*.txt` builds the deployed image) plus its own requirements:
`numpy`, `torch`, `transformers`, `lightgbm`, `scikit-learn`.

---

## Site 1 — `app/services/variable_registry.py:74` `normalize_field`

**Current**: a 40-entry dict lookup. Returns `None` for anything not in the
dict (which is most real provider field names — see `docs/ML_PIPELINE.md`).

**Add**: when the dict misses, fall back to M1.

```python
def normalize_field(raw_field: str, *, unit: str | None = None,
                    description: str = "") -> dict | None:
    key = raw_field.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    key2 = key.replace(" ", "_").replace("-", "_")
    if key2 in REGISTRY:
        return REGISTRY[key2]
    if _registry and _registry.field_mapper is not None:
        mapping = _registry.field_mapper.map_field(raw_field, unit=unit,
                                                    description=description)
        if mapping.is_usable:
            return {"canonical": mapping.canonical_variable,
                   "statistic": mapping.statistic,
                   "accumulation_hours": ([mapping.accumulation_hours]
                                          if mapping.accumulation_hours else []),
                   "evidence_class": [mapping.evidence_class],
                   "_algorithm_version": mapping.algorithm_version,
                   "_confidence": mapping.confidence}
    return None
```

Every decoder that calls `normalize_field` (`app/decoders/*.py`) already
handles a `None` return by not fabricating a mapping — that behavior is
unchanged, just the fallback got smarter. When the returned dict carries
`_algorithm_version`, propagate it into the CEO's `Provenance.transformations`
list so the field mapper's involvement is visible in the evidence chain.

---

## Site 2 — a new derived-CEO step, after `filter_by_window` in `_weather_request`

**Current** (`app/main.py:96-120`): raw adapter output goes straight to
`validated_evidence` and `build_wio`. Nothing corrects bias, nothing attaches
a calibrated probability.

**Add**: for every forecast-class CEO carrying `precipitation_amount`,
`temperature_2m`, or `wind_speed_10m` from a source in `{gfs_seamless,
ecmwf_ifs025, icon_seamless, gem_seamless}`, emit a **derived CEO** from M2
alongside the original (never replacing it — the CEO schema's whole point is
that a correction is provenance-linked to what it corrected, not a silent
overwrite).

```python
def _apply_mos_correction(evidence: list, registry: ModelRegistry) -> list:
    if registry is None or registry.mos is None:
        return evidence
    by_key: dict[tuple, dict[str, float]] = {}
    for item in evidence:
        if item.variable in ("precipitation_amount", "temperature_2m", "wind_speed_10m") \
                and item.model_name in MODELS and item.evidence_class == "forecast":
            var = {"precipitation_amount": "precipitation",
                  "temperature_2m": "temperature_2m",
                  "wind_speed_10m": "wind_speed_10m"}[item.variable]
            key = (item.geometry.coordinates, item.valid_from, var)
            by_key.setdefault(key, {})[item.model_name] = item.value

    derived = []
    for (coordinates, valid_from, var), forecasts in by_key.items():
        if len(forecasts) < 2 or not registry.mos.supports(var):
            continue  # too few models agreeing to trust a correction
        corrected = registry.mos.correct(var, forecasts=forecasts, context={
            "lead_hours": ..., "lead_age_days": ..., "hour_utc": valid_from.hour,
            "doy": valid_from.timetuple().tm_yday, "elevation_m": ...,
            "lat": coordinates[1], "lon": coordinates[0],
        })
        parent_ids = [item.evidence_id for item in evidence
                      if item.model_name in corrected.parents
                      and item.geometry.coordinates == coordinates]
        derived.append(CanonicalEvidenceObject(
            source="OTHER", evidence_class="forecast",
            variable={"precipitation": "precipitation_amount",
                     "temperature_2m": "temperature_2m",
                     "wind_speed_10m": "wind_speed_10m"}[var],
            value=corrected.value, unit=..., statistic="instant",
            geometry=Geometry(coordinates=coordinates), valid_from=valid_from,
            model_name="weathergpt_mos_v1", parent_ids=parent_ids,
            transformation=corrected.algorithm_version,
            transformation_timestamp=datetime.now(timezone.utc),
            extra={"raw_ensemble_mean": corrected.raw_ensemble_mean,
                  "quantiles": corrected.quantiles,
                  "interval_80": [corrected.interval_low, corrected.interval_high]},
            provenance=Provenance(original_source="weathergpt_mos_v1",
                                 transformations=["MOS bias correction"])))
    return evidence + derived
```

(Sketch — you'll need to fill in `lead_hours`/`lead_age_days`/`elevation_m`
from whatever the adapter already resolved, and the exact CEO field names may
need adjusting to match `app/schemas/ceo.py` precisely.) The key invariant:
**`parent_ids` must point at real evidence IDs already in the store**, so the
reviewer agent (`app/agents/orchestrator.py:62-75`) can still verify the
derived claim traces back to real sources.

`wio_builder.build_wio` should prefer the derived (`weathergpt_mos_v1`) CEO
over the raw one when both exist for the same variable/location/time — that's
a one-line change to whatever "pick best per variable" logic it already has
(`app/services/wio_builder.py:8-101`).

---

## Site 3 — `app/rade/v2.py:generate_scenarios`

**Current** (`rade/v2.py:66-81`): bins raw ensemble member values into five
precipitation scenarios, or falls back to a provider's point probability.

**Add**: replace member-counting with M4's calibrated exceedance curve.

```python
def generate_scenarios(wio, registry: ModelRegistry | None = None):
    if registry and registry.calibration is not None:
        precip = wio.weather.rain or {}
        forecasts = precip.get("member_forecasts_by_model")  # you'll need to
                                                              # carry this through
                                                              # from the CEO
        if forecasts:
            curve = registry.calibration.probability_curve(
                "precipitation", forecasts=forecasts, context={...})
            scenarios = [Scenario(
                name=f"{lo}-{hi}mm",
                probability=curve[i].probability - (curve[i+1].probability
                                                    if i+1 < len(curve) else 0),
                precipitation_mm=(lo + hi) / 2, evidence_ids=evidence_ids)
                for i, (lo, hi) in enumerate(zip(
                    [0] + list(registry.calibration.thresholds),
                    list(registry.calibration.thresholds) + [999]))]
            return scenarios, ["Scenarios use calibrated exceedance probabilities, "
                              "verified Brier-optimal at each threshold."]
    # fall through to the existing member-binning / point-probability logic
    ...
```

This is the single highest-value integration point: **RADE's whole
risk-adjusted-decision story is only as honest as the probabilities feeding
it**, and right now those probabilities are either raw member counts (known
to be over/under-confident — that's the entire reason M4 exists) or a single
provider point estimate.

---

## Site 4 — `app/services/ranker.py:8-50`

**Current**: `score = 0.4*authority + 0.25*freshness + 0.20*spatial +
0.15*quality`, with a static per-source `AUTHORITY` table — necessarily one
global ordering regardless of location, season, or lead time (this is exactly
what M5 was built to test and beat).

**Add**: when ranking among the four NWP-model sources for a single variable,
call M5 instead of the static table.

```python
def rank_sources(candidates: dict[str, float], variable: str, context: dict,
                 registry: ModelRegistry | None = None) -> list[str]:
    if registry and registry.trust_ranker is not None and registry.trust_ranker.supports(variable):
        ranked = registry.trust_ranker.rank(variable, candidates=candidates, context=context)
        return [r.source for r in ranked]
    return _rank_by_static_authority(candidates)  # existing logic, kept as fallback
```

Only applies to the four NWP-model sources M5 was trained on
(`gfs_seamless`, `ecmwf_ifs025`, `icon_seamless`, `gem_seamless`); IMD, CAP and
other non-NWP sources keep the existing authority-table path since M5 was
never trained on them.

---

## Site 5 — `app/orchestrator/retrieval_planner.py` + `app/services/time_parser.py`

**Current**: keyword/regex-based intent and time-window extraction.

**Add**: run M3 alongside the rule-based parser, not instead of it — the rule
parser is the floor, M3 augments it.

```python
def build_retrieval_plan(question: str, horizon: str,
                         registry: ModelRegistry | None = None) -> RetrievalPlan:
    plan = _build_retrieval_plan_rule_based(question, horizon)  # existing function
    if registry and registry.intent is not None:
        parsed = registry.intent.parse(question)
        if parsed.intent_confidence > 0.35:
            plan.decision_context = plan.decision_context or parsed.intent
        # do NOT use parsed.variables -- M3's variable head is a known weak
        # point (micro-F1 0.170 on held-out data as of the v2 retrain); only
        # intent and slots (parsed.slot_text("LOC")/"TIME"/"CROP") are
        # validated for use
    return plan
```

**The 0.5 confidence threshold in an earlier draft of this doc does not
work — corrected here after live testing, not from theory.**
`modal_jobs/realworld_check.py` ran M3 against a dozen genuinely
never-templated queries (casual phrasing, typos, "tmrw"/"rn"-style
shorthand, code-switched Hindi/Tamil, out-of-domain questions) and every
single one came back with `intent_confidence` between 0.148 and 0.19 — for
an 8-class softmax, that is barely above the ~0.125 uniform-guess floor. A
`> 0.5` gate means M3's intent output would **never fire on real user
phrasing**, only on text that looks like the D4 templates it was trained on.
0.35 is not a validated threshold either (there is no labelled real-world
eval set to tune it against yet) — it is a documented placeholder that lets
the signal through as a tie-breaker while the rule parser still owns the
decision when both disagree or when M3 abstains below the gate. Whoever
wires this in should treat the threshold as a knob to revisit once real
query logs exist, not as settled.

Also observed in that same test: intent *accuracy* on natural phrasing is
mixed even when confidence is set aside. "should i take umbrella tmrw
morning shillong" was classified `harvest` (should be a plain rain query);
"hey can u tell me if its safe for boats near puri tomorow morning" came
back `travel` where `marine` was the better fit (M3 does get near-identical
phrasing right on template-style input — see `verify_package.py`'s
"Can fishermen go to sea off Ratnagiri tomorrow?" -> `marine`). **Location
and time slot extraction held up much better** on the same live queries —
`LOC`/`TIME` spans were extracted correctly on 10 of 12, including through
typos and shorthand — which is why this doc's original guidance to trust
slots but not variables is extended here to *also* not over-trust the raw
intent label on organic phrasing without the rule parser as the deciding
vote.

**Do not call `registry.intent`'s `variables` output for anything.** It is
documented in `backup/MODEL_RESULTS_LOG.md` as broken (macro-F1 0.120 on
held-out data even after the v2 retrain — most of the 46 canonical
variables still have thin or zero support). Location and time resolution
should still go through `location_resolver.py` and `time_parser.py` — M3
gives you the *substring* to feed them, not a resolved coordinate or
timestamp.

---

## What NOT to change

- **Never let a model's absence break a request.** Every site above has a
  `registry is None or registry.X is None` fallback to the exact logic that
  runs today. The five-model gate in `weathergpt_models/registry.py` already
  refuses a bad artifact at load time; the serving code's job is to handle
  that refusal gracefully, not to assume success.
- **Never present a corrected/calibrated value as if it came directly from a
  provider.** Always emit it as a derived CEO with `parent_ids`,
  `transformation`, and `algorithm_version` set — that's what lets the
  reviewer agent keep doing its job, and what lets a user's "why do you say
  that" drill down to the real answer.
- **M3's location/time spans are substrings, not resolved values.** Still run
  them through `location_resolver.resolve_location` and
  `time_parser.parse_time_window`. Never fabricate coordinates from a model's
  own confidence.

---

## Verification once wired

1. `pytest tests/` — the existing `test_api_contracts.py`,
   `test_location.py` etc. should still pass unchanged; add new ones for each
   integration site (mock `registry` as `None` to confirm the fallback path,
   then with a real loaded registry to confirm the enhanced path).
2. Live: `GET /health` should report each model's load status (add a
   `"models": registry.status()` block, replacing the current
   `"rule-based fallback only"` placeholder at `app/main.py:150`).
3. `POST /query` for a district in M3's training location set vs one outside
   it — confirm the fallback path degrades gracefully for the latter rather
   than returning a wrong-but-confident intent.
