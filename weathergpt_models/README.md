# `weathergpt_models` — trained model layer

Five models, each with measured metrics and a baseline it had to beat. This
package is the **inference interface**; the framework consumes it and never
imports the training stack.

```python
from weathergpt_models import ModelRegistry

registry = ModelRegistry.from_hub("<repo-id>")   # or .from_dir("/path/to/artifacts")

for row in registry.status():
    print(row)          # which models loaded, and for any that did not, exactly why
```

## Dependencies

`numpy`, `torch`, `transformers`, `lightgbm`, `scikit-learn`, and
`huggingface_hub` only if you use `from_hub`. No FastAPI, no pydantic, no
coupling to `app/`.

Everything runs on CPU. Pass `device="cuda"` if you have a GPU.

---

## The admission gate

An artifact only serves if its own `metrics.json` proves three things:
provenance (dataset kind, SHA-256, split, timestamp), a baseline it was measured
against, and a margin over that baseline. Anything else is refused, the
attribute stays `None`, and `status()` names the failing gate.

```python
registry.field_mapper is None      # refused -> use your deterministic path
```

This exists for a concrete reason: this repository previously shipped a
`best.pt` fitted to `np.random.randn(500, 5)` for one epoch, alongside a
`metrics.json` declaring `"dataset_kind": "real_matched_pairs"`, and nothing in
the system could tell. Now it can.

---

## M1 — field mapper

*What does this native field name mean?* Maps an arbitrary provider field onto
the canonical vocabulary, or refuses.

```python
m = registry.field_mapper.map_field(
    "APCP", unit="kg m-2", time_range_text="0-3 hour acc fcst")

m.canonical_variable   # "precipitation_amount"
m.statistic            # "accumulation"
m.accumulation_hours   # 3.0
m.abstained            # False
m.source               # "rule" | "model" | "abstain"
```

The deterministic taxonomy runs first; the trained model handles what it
abstains on — schemas nobody has written rules for. Both can refuse.
`m.is_usable` is the single check for "did we actually get a mapping".

**Do not treat an abstention as `precipitation_amount` because the name looks
like rain.** That is the failure mode the whole project exists to prevent.

## M3 — intent parser

*Which decision is being asked about, and which spans mean what?* Thirteen
Indian languages and scripts, including romanised Hinglish and Banglish.

```python
q = registry.intent.parse("kal Bhandara me baarish hogi kya?")

q.intent            # one of: none, spray, irrigate, harvest, marine, travel, sow, warning_check
q.variables         # canonical variables to retrieve — use this to prune the retrieval plan
q.slot_text("LOC")  # "Bhandara"      -> hand to the geocoder
q.slot_text("TIME") # "kal"           -> hand to the time parser
q.slot_text("CROP") # None
```

It returns **substrings, not interpretations**. Resolving "Bhandara" to
coordinates and "kal" to a UTC window stays the framework's job.

## M2 — MOS corrector

*The raw NWP value is biased; here is the corrected one, with an interval.*

```python
c = registry.mos.correct(
    "temperature_2m",
    forecasts={"gfs_seamless": 31.2, "ecmwf_ifs025": 30.4,
               "icon_seamless": 30.9, "gem_seamless": 31.8},
    context={"lead_hours": 30, "lead_age_days": 1, "hour_utc": 6, "doy": 245,
             "elevation_m": 303, "lat": 21.15, "lon": 79.09},
    other_forecasts={"relative_humidity_2m": {...}, "precipitation": {...}})

c.value           # corrected median
c.interval_low, c.interval_high   # conformalised 80% interval
c.correction      # how far it moved the raw ensemble mean
c.parents         # which sources fed it — for your evidence lineage
```

Missing models are fine: pass what you have, the rest are imputed and flagged.
At least two of the four should be present for the spread features to mean
anything.

## M4 — probability calibrator

*How likely is this event, honestly?* This is what a decision engine should
consume — never a raw member count.

```python
p = registry.calibration.exceedance_probability(
    "precipitation", 5.0, forecasts={...}, context={...})

p.probability             # calibrated
p.raw_ensemble_frequency  # what counting members would have said
p.method                  # "csgd_isotonic"

registry.calibration.probability_curve("precipitation", forecasts=..., context=...)
# every verified threshold: 0.1, 1, 5, 10, 25, 50 mm
```

Reliability curves and Brier skill scores per threshold are in `metrics.json`,
measured on locations the model never trained on.

`registry.calibration.transfer_assumption` states, in prose, the one assumption
this model rests on. Read it before quoting the probabilities anywhere official.

## M5 — trust ranker

*Which source should be believed here, at this lead, in this season?*

```python
ranked = registry.trust_ranker.rank(
    "temperature_2m",
    candidates={"gfs_seamless": 31.2, "ecmwf_ifs025": 30.4, ...},
    context={"lead_hours": 30, "lead_age_days": 1, "doy": 245, "month": 9,
             "elevation_m": 303, "lat": 21.15, "lon": 79.09})

ranked[0].source, ranked[0].score, ranked[0].value
```

Replaces a static authority table, which by construction produces one global
ordering — the same model winning in Leh in January and Kochi in July.

---

## Provenance for your evidence objects

Every result carries `algorithm_version`, and M2/M4 carry `parents` (the source
models that fed them). Emit them as derived evidence with those fields set, so a
corrected value can always be traced back to the raw records it came from.

## Reading the numbers

`registry.metrics("mos")` returns the full `metrics.json`. Every headline figure
sits next to the baseline it was measured against, stratified by lead time,
region and season, on a test set that holds out **both** future time **and**
locations never trained on. The held-out-location split is the one that matters
for a user asking about a district that was not in the corpus.
