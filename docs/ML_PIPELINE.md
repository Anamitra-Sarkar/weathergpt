# The WeatherGPT model layer

Five models, four training corpora, one inference package. Everything is built
and trained on Modal; nothing is downloaded to a developer machine.

```
                    Open-Meteo archives          authoritative parameter tables
                    (4 NWP models + ERA5)        (CF, GRIB2, NCEP, WRF, BUFR, CAP)
                            |                                   |
                            v                                   v
   D1  multi-model MOS corpus                       D3  field-name corpus
   127 locations x 12 months x 8 lead ages          15,243 rows, 44 classes
        |            |            |                          |
        v            v            v                          v
   M2 MOS      M4 calibration   M5 trust ranker         M1 field mapper
        \            |            /                          |
         `-----------+-----------'                           |
                     |                                       |
              D4 multilingual queries  ------->  M3 intent + slot parser
                     |                                       |
                     `------------------+--------------------'
                                        v
                              weathergpt_models
                        (registry + admission gate)
```

---

## The corpora

### D1 — multi-model MOS corpus  `modal_jobs/build_corpora.py --what d1`

For each of 127 Indian locations, twelve months of hourly forecasts from four
independent NWP models (GFS, ECMWF IFS, ICON, GEM) at eight forecast lead ages,
verified against ERA5 reanalysis.

The lead ages come from Open-Meteo's `<variable>_previous_dayN` parameters,
which return the forecast for a valid time as it was issued *N days earlier*.
That makes `lead_hours = 24N + hour_utc` a real forecast lead. The previous
pipeline used `lead_hours = i % 72` where `i` was the row index, so that feature
was noise; `contracts.check_lead_time_signal` now asserts that mean error
actually grows with lead, and would catch a regression.

Two things about this fetch cost a rebuild each and are worth not rediscovering:

- **`era5_land` publishes no precipitation or wind through this API** — 100%
  nulls. `era5_seamless` keeps ERA5-Land's finer temperature and fills the rest
  from ERA5 proper.
- **Open-Meteo answers a 128-column request with a 502.** One model per request,
  46-day chunks: 200, 181 KB, 2.3 s.

Locations are resolved through the Open-Meteo geocoding API, so latitude,
longitude, elevation and the administrative hierarchy all come from the
provider. The previous pipeline hardcoded twenty elevation literals, which made
elevation perfectly collinear with location and unlearnable.

Shards commit per location and skip what already exists, so an interrupted run
resumes instead of starting over.

### D2 — ensemble corpus  `--what d2`

Built, then found unusable, and the finding is worth keeping. The ensemble API
accepts `past_days=93` and returns a well-formed response, but the member
columns are null for everything except roughly the last four days. ERA5 truth
lags about six days. The two windows do not overlap, so a verified 31-member
training corpus cannot be built from this source at all.

M4 therefore trains on D1's four-model ensemble and its model card states the
transfer assumption rather than hiding it.

### D3 — field-name corpus  `modal_jobs/build_fields.py`

15,243 native field names harvested from the standards that actually define
them: the CF standard name table, the ECMWF eccodes GRIB2 definitions, NOAA
NCEP GRIB2 code tables joined to real GFS `.idx` inventories, the WRF Registry,
the WMO BUFR element table, Open-Meteo's own variable names, IMD product fields,
and the live SACHET/NDMA CAP feed for official warning events.

No string augmentation. Each row is labelled by `weathergpt_models/taxonomy.py`
from **that standard's own metadata** — declared unit, level type, statistical
processing code, time-range string — never from the abbreviation. Two
consequences:

- `TMAX (mm)` cannot exist. The labeller abstains on a unit contradiction. The
  previous corpus had 64 temperature rows carrying a depth unit because the
  suffix was sampled without consulting the label.
- Accumulation windows are read out of real inventory strings
  (`0-3 hour acc fcst`) instead of being sampled from `{1, 3, 6, 24}`.

**The split is by source table**: train on CF + GRIB2 + CAP, test on WRF, NCEP,
BUFR, Open-Meteo and IMD. That measures generalisation to a schema nobody has
written rules for, which is the thing the product actually needs.

### D4 — multilingual query corpus  `modal_jobs/build_queries.py`

Templated questions instantiated with real district names and real crops, then
translated into thirteen Indian languages and scripts by Groq — including
romanised Hinglish and Banglish.

Slot labels are exact by construction: the builder knows the substrings it
injected, so the character spans and therefore the BIO tags are known, not
inferred. For the translations the model must return the translated slot values
as JSON alongside the sentence, and **any row whose returned slot is not
literally a substring of the returned sentence is discarded**. A paraphrase
cannot silently corrupt a label.

The predecessor assigned `decision = random.choice(...)` for about two thirds of
its rows, making the label statistically independent of the text, and hardcoded
`location: "Nagpur", time: "tomorrow"` on every Groq paraphrase.

The split holds out whole template families **and** whole districts, so neither
a memorised sentence pattern nor a memorised place name can inflate the score.

---

## The models

| | what it decides | trained on | beats |
|---|---|---|---|
| **M1** field mapper | what a native field name means | D3 | the dict registry in `variable_registry.py` |
| **M2** MOS | the corrected value and its interval | D1 | raw NWP, multi-model mean, persistence, climatology |
| **M3** intent parser | which decision is being asked, and which spans mean what | D4 | the rule-based retrieval planner |
| **M4** calibration | how likely an event actually is | D1 | counting ensemble members, and climatology |
| **M5** trust ranker | which source to believe here and now | D1 | the fixed authority order in `ranker.py` |

**M1** is a label-embedding bi-encoder rather than a softmax classifier. The
official-warning classes have single-digit support — IMD's real CAP event
vocabulary is small — and a softmax row learned from three examples is
worthless, whereas scoring against the *text* of the label transfers meaning
from the pretrained encoder. A new canonical variable can be added later by
writing one sentence.

**M2** emits seven monotone quantiles trained with pinball loss, anchored on the
raw ensemble mean rather than predicted from scratch — the output layer
initialises at zero, so training starts as the identity correction and spends
its capacity on the part that is actually hard. A network and a gradient-boosted
forest are both fitted; which one is served (or their blend, at a weight chosen
on validation) is *also* selected on validation, never on the held-out test
split, so the reported test metric stays an honest estimate of what a fresh
deployment would see. Precipitation is fitted in cube-root space and cubed
back — exact, because quantiles are equivariant under a monotone transform.

**M3** is a JointBERT head on MuRIL: intent, BIO slot spans and multi-label
variables from one encoder. MuRIL was pretrained on 17 Indian languages *and*
their transliterations, so Hinglish is in-distribution. The predecessor used
`distilbert-base-uncased`, which turns every Devanagari character into `[UNK]`.

**M4** fits a hurdle model for precipitation only: an explicit P(wet) times a
censored shifted gamma for how much falls if it is — a point mass at zero for
the dry hours, a long right tail for monsoon bursts — by minimising CRPS, then
refines each exceedance threshold with isotonic regression. Two implementation
notes that matter: quadrature of the predictive CDF is the obvious approach and
it is wrong here, because a CSGD fitted to mostly-dry precipitation converges to
a very small shape parameter where the gamma density has an integrable
singularity at zero that no polynomial grid resolves; and
`torch.special.gammainc` is exact but carries no gradient in the shape. Training
estimates CRPS from reparameterised `Gamma.rsample` draws; inference uses
`gammainc`. Temperature and wind were tried too and dropped: anchored the same
way, they still lost to the raw four-model ensemble's fair CRPS by -37.9% and
-24.7%, because a discrete four-point ensemble from genuinely skillful models is
a hard baseline for a symmetric two-parameter distribution to beat, and M2's
quantile network already beats it on both. M4 is scoped to the exceedance
probabilities it actually wins at.

**M5** exists to test a claim that was previously untested. The authority terms
in `app/services/ranker.py` are constant across sources at a given point and
time, so the formula collapses to one global ordering — the same model winning
in Leh in January and in Kochi in July. LambdaMART conditions on lead, season,
elevation, spread and each source's measured skill at that location, and is
scored on whether following it actually lands closer to truth.

---

## Running it

```bash
# corpora
modal run --detach modal_jobs/build_corpora.py --what locations
modal run --detach modal_jobs/build_corpora.py --what d1 --n-shards 8
modal run --detach modal_jobs/build_fields.py
modal run --detach modal_jobs/build_queries.py

# models
modal run --detach modal_jobs/train_field_mapper.py --epochs 8
modal run --detach modal_jobs/train_mos.py --epochs 30
modal run --detach modal_jobs/train_calibration.py --epochs 40
modal run --detach modal_jobs/train_intent.py --epochs 8
modal run --detach modal_jobs/train_trust_ranker.py

# publish
modal run modal_jobs/export_models.py --repo-id <user>/<name>
```

`backup/wait_and_train.sh` waits for D1 to finish and then launches the three
trainers that depend on it.

A few operational notes:

- Modal images must list every local package in `.add_local_python_source(...)`
  or remote functions die at import.
- A remote function parameter cannot be annotated `list[str] | None` or `list`;
  Modal cannot parse either.
- `modal run --detach` still dies if the launching process is killed during the
  initial run phase. Launch through a harness-managed background job, or
  `setsid`.
- `max_containers` is capped at 8 because other work shares this Modal
  workspace.

## Verification

```bash
pytest tests/test_model_package.py -q
```

Covers the properties that previously failed silently: a label that contradicts
its own unit, an imputed ensemble member reaching the model unflagged, a CDF
that loses mass, a quantile transform applied where it is not valid, and an
artifact whose metrics nobody checks.

Every trainer runs `modal_jobs/contracts.py` before its first gradient step. The
headline contract is four lines and would have caught the single worst defect in
the previous pipeline: `max |forecast - truth| > 1e-3`. The shipped
`matched_pairs.csv` failed it — forecast and observation were the same series,
so the regression target was zero everywhere and the model reported an excellent
RMSE for having learned nothing.
