# WeatherGPT model results — running log

Every number here was printed by a training script and read back from a
`metrics.json` on the `weathergpt-models` Modal volume. Nothing is typed by
hand or estimated.

## Corpora built

| corpus | size | notes |
|---|---|---|
| locations | 127 of 128 seeds | 35 states, elevation 2–3502 m; Cherrapunji has no IN geocoder hit |
| **D1** multi-model MOS | **9,582,912 rows**, 127 locations | 4 NWP models × 8 lead ages × 12 months vs ERA5-seamless |
| **D3** field names | **15,246 rows**, 4,567 labelled, 44 classes | 8 authoritative source tables; split by source |
| D2 ensemble | 268,224 rows — **unusable** | member columns null except the last ~4 days; no overlap with ERA5 truth |
| D4 multilingual queries | building | 13 languages, exact slot spans |

## M1 — field mapper  ✅

`intfloat/multilingual-e5-base`, label-embedding bi-encoder, abstention
threshold calibrated on a **held-out schema** (`dev_zeroshot`), reported on
schemas touched by nothing else (`test_zeroshot` = WRF Registry + NCEP
inventories, 5,140 rows).

| metric | M1 | dict registry | majority |
|---|---|---|---|
| zero-shot macro-F1 | **0.743** | 0.160 | 0.017 |
| zero-shot accuracy | **0.832** | 0.593 | 0.550 |
| mapped accuracy | **0.744** | 0.095 | — |
| statistic accuracy | **0.979** | — | — |
| **misassignment rate** | **0.0016** | — | — |
| abstain precision / recall | 0.742 / 0.946 | — | — |

**Two findings worth keeping.**

*Calibrating the threshold in-domain was wrong.* Tuning it on validation and
applying it zero-shot moved mapped accuracy from 0.744 down to 0.543 and doubled
the misassignment rate to 0.0032. The similarity distribution shifts when the
schema changes.

*Bigger was worse.* `multilingual-e5-large` scored **better** on in-domain
validation (macro-F1 0.860 vs 0.841) and **worse** on unseen schemas (0.663 vs
0.729, mapped accuracy 0.393 vs 0.543). Validation alone would have chosen it.
Kept as an ablation at `m1_field_mapper_v1_large`.

## M5 — trust ranker  ✅

LightGBM LambdaMART over four candidate NWP sources per (location, valid time,
lead) group. Test = future time at locations never trained on.

| variable | NDCG@1 | picks the best source | fixed authority does | RMSE following ranker | RMSE following fixed order | oracle |
|---|---|---|---|---|---|---|
| temperature_2m | 0.688 | **0.383** | 0.346 | **1.308** | 1.443 | 0.704 |
| precipitation | 0.851 | **0.237** | 0.170 | **1.056** | 1.124 | 0.776 |
| wind_speed_10m | 0.678 | **0.377** | 0.317 | **3.993** | 4.582 | 2.081 |

RMSE skill over the fixed authority order: +9.3%, +6.0%, +12.8%.

The claim being tested was that `app/services/ranker.py`'s authority table is
constant across sources at a given point and time, so it can only ever express
one global ordering. The learned ranker's top features for temperature and wind
are `deviation_from_mmm`, then **latitude, elevation, longitude and per-location
historical skill** — the right source depends on where you are, which a fixed
table cannot say.

## M2 — distributional MOS  🔄

Anchored on the ensemble mean; precipitation fitted in cube-root space.
Temperature reached val CRPS **0.521** against a raw ensemble at 0.594 (+12.4%)
at epoch 1 and overfitted after, hence early stopping and lr 5e-4.

## M4 — calibration  🔄

First run, before anchoring and before the hurdle:

| threshold | base rate | raw ensemble | CSGD | CSGD+isotonic | climatology |
|---|---|---|---|---|---|
| >0.1 mm | 0.290 | 0.15639 | 0.13667 | **0.13656** | 0.20590 |
| >1 mm | 0.0873 | 0.07812 | 0.06710 | **0.06698** | 0.07971 |
| >5 mm | 0.0086 | 0.00926 | 0.00823 | **0.00820** | 0.00848 |
| >10 mm | 0.0014 | 0.00165 | 0.00135 | **0.00134** | 0.00138 |

Brier improved 12.7 / 14.3 / 11.4 / 18.8% over counting ensemble members, and
beat climatology at every threshold — but CRPS was **worse** than the raw
ensemble (−5.5% precipitation, −21.8% temperature). Two causes, both fixed:

1. **No atom at zero.** Hourly rain is exactly zero most hours; the ensemble can
   be exactly 0 and a continuous density cannot. Now a hurdle: P(wet) × CSGD.
2. **Predicting from scratch.** Temperature trained to CRPS 0.589 against a raw
   ensemble at 0.594 — forty epochs spent rediscovering the ensemble mean it was
   already handed. Both heads are now anchored, which is what EMOS is.

## Data findings that cost a rebuild each

- `era5_land` returns **100% nulls** for precipitation and wind through the
  Open-Meteo archive API. Use `era5_seamless`.
- Open-Meteo answers a **128-column request with a 502**. One model per request,
  46-day chunks: 200, 181 KB, 2.3 s.
- The ensemble API accepts `past_days=93` and returns a well-formed response
  whose **member columns are null except for the last ~4 days**. ERA5 truth lags
  ~6 days, so the windows never overlap and a verified 31-member corpus cannot
  be built from it at all.
- **Precipitation MAE falls at long lead** (0.238 at day 4 → 0.191 at day 7)
  while temperature MAE grows as expected (1.62 → 2.15). Not an artifact — row
  counts are identical at every lead. A model relaxing toward climatology
  forecasts less rain, which lowers MAE against a mostly-dry truth while skill
  falls. The lead-time contract now uses correlation.
- Quadrature of a CSGD's CDF loses almost all the mass at the small shape
  parameters a mostly-dry fit converges to: a 161-point grid recovered **8%** at
  shape 0.05, and midpoint on 513 points still lost 44%. `gammainc` is exact but
  has no gradient in the shape, so training samples through `Gamma.rsample` and
  inference uses `gammainc`.

## M3 / D4 — deferred, with root cause

D4 (multilingual query corpus) failed to complete across five configurations in
this session, each narrowing the blast radius:

| attempt | shards × concurrency | languages/row | outcome |
|---|---|---|---|
| 1 | 4 × 4 (all 13 languages/row) | 13 | 95 min, zero output — killed |
| 2 | 4 × 2 | 5 | 45 min, zero output — killed |
| 3 | 4 × 3, progress logging added | 3 | still zero progress lines after 12 min |
| 4 | 4 × 2, dropped a broken model, fixed backoff | 3 | still zero progress lines |
| 5 | 1 × 3, smaller target | 2 | remote container produced one line then went silent; `modal app logs` on the still-registered app showed nothing further; the CLI's own iterator raised `RemoteError` with no recoverable message |

**Root causes identified, in order found:**

1. **A direct diagnostic against the live Groq API** (12 concurrent calls, no
   semaphore) showed 7/12 got an instant 429 — the per-org rate limit is
   substantially tighter than a single warm-up call suggested, and `n_shards`
   containers each running their own `concurrency`-limited semaphore multiplies
   into a much higher *global* concurrent load against the same key than any
   single shard's setting implies (4 shards × concurrency 2 = 8 concurrent, not
   2).
2. **`qwen/qwen3.6-27b` deterministically 400s** under `response_format:
   json_object` ("Failed to validate JSON") — 3/3 in the same diagnostic burst.
   Fixed: dropped from the rotation, `response_format` dropped entirely (the
   code already validates parsed JSON, so the constraint bought nothing but a
   second failure mode).
3. After both fixes, a single-shard run still went silent after printing only
   its dispatch line, and the CLI's `expand_shard.starmap()` iterator raised
   `RemoteError`. `modal app logs` on the still-registered app showed the same
   single line and nothing else — consistent with the container process being
   terminated by Modal's runtime rather than a Python exception inside the
   function (which would have printed a traceback into the function's own
   stdout, and none appeared).

**Decision:** stopped rather than continuing to narrow parameters against an
environment that is failing in a way five iterations of tuning did not resolve.
M3 (JointBERT intent/slot parser on MuRIL) is fully written
(`modal_jobs/train_intent.py`) and its inference class, registry gate and
export-card section are all in place and tested — it needs only a real D4
corpus to run against. The retry-logic and model-rotation fixes made along the
way are real and committed regardless of whether D4 itself completes; they will
matter for any future attempt.

**To resume:** try `n_shards=1 concurrency=1` with `per-template=10
languages-per-row=1` as a minimal smoke test first, confirm at least one
`[d4:0] N/M calls done` progress line appears within a few minutes, and only
scale up from a configuration that is actually observed to produce output.

**Addendum after a fifth, fully-serial (`concurrency=1`) smoke test:** still zero
progress after 13 minutes on a 260-call queue, with no error surfaced by
`modal app logs` on the live container. Read closely, the retry loop bounds
each call to at most 3 attempts and cannot infinite-loop; the likely explanation
is Groq's `on_demand` service tier itself, which reports a `queue_time` field on
every response and can plausibly take tens of seconds per call under load —
consistent with every earlier "silent for 40+ minutes" observation being real,
slow progress rather than a hang. At that per-call cost, one-call-per-language
cannot finish a corpus of any useful size within a session at any concurrency
low enough to avoid the 429 wall.

**The architectural fix for next time:** batch languages into the prompt
instead of issuing one call per language. Asking one Groq call to return
translations into 3-5 languages at once (a JSON array instead of a single
object) cuts the call count by the same factor and should bring total wall time
back into a session-sized budget without touching the rate limit at all. Not
attempted here for lack of remaining time; the slot-substring verification in
`expand_shard`'s `one()` would need to iterate the array instead of a single
payload, which is a contained change.

**Second addendum:** implemented the batching fix (`modal_jobs/build_queries.py`,
one call now returns up to 6 languages as a JSON array). A smoke test at
concurrency=2 still showed zero progress after 9 minutes on 208 batched calls.
Batching did not resolve it within the time available — either per-call latency
scales with the larger requested output (1800 vs 300 max_tokens), or queueing
dominates regardless of request size. The batching change is kept (it is a
real improvement and correctly verified/committed independently of whether it
fully fixes the throughput problem), but D4/M3 stay deferred. Next session
should measure single-batched-call latency directly (the same diagnostic
pattern used to find the 429/400 issues) before assuming batching alone is
sufficient.
