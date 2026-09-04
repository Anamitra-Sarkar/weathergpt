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
