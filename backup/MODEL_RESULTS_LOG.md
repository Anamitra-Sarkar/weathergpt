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
| D4 multilingual queries | **deferred** — see below | 13 languages, exact slot spans; blocked on Groq on_demand-tier throughput |

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

## M2 — distributional MOS  ✅

Anchored on the ensemble mean (both the quantile network and the LightGBM
forest predict a correction, not an absolute value); precipitation fitted in
cube-root space; head served (net / GBM / their blend) chosen on validation,
never on the reported test split.

| variable | CRPS (model) | CRPS (raw ensemble) | CRPS skill | served head |
|---|---|---|---|---|
| temperature_2m | 0.616 | 0.647 | **+4.7%** | blend (w=0.75) |
| precipitation | 0.219 | 0.252 | **+13.2%** | blend (w=0.80) |
| wind_speed_10m | 1.600 | 1.757 | **+8.9%** | blend (w=0.60) |

The blend of the quantile network and LightGBM beat both components alone on
every variable, confirming they make different mistakes (the network
extrapolates smoothly, the trees capture sharp terrain/regime splits) worth
combining. RMSE also beats the raw GFS point forecast substantially — e.g.
wind 3.60 vs 7.35 m/s — and the multi-model mean (wind 3.60 vs 3.72).

## M4 — calibration  ✅ (precipitation only)

Rescoped after temperature and wind, even anchored, still lost to the raw
ensemble's fair CRPS (-37.9%, -24.7%) -- M2 already wins there and is the
served corrector for those two variables. Final run, anchored, with the hurdle
(P(wet) x censored shifted gamma):

**CRPS** (not the model's job, but measured anyway): test 0.249 vs raw
ensemble 0.232, CRPSS **-7.6%** -- a discrete four-point ensemble from
genuinely skillful models remains a hard baseline to beat on point/distributional
accuracy, which is exactly why M4 is scoped to what it does win at.

**Exceedance Brier score** (M4's actual job — this is what a decision engine
should read, never a raw member count):

| threshold | base rate | raw member count | CSGD | CSGD+isotonic | climatology |
|---|---|---|---|---|---|
| >0.1 mm | 0.316 | 0.16660 | 0.14877 | **0.14883** | 0.21616 |
| >1 mm | 0.093 | 0.08376 | 0.07281 | **0.07243** | 0.08455 |
| >5 mm | 0.008 | 0.00909 | 0.00809 | **0.00807** | 0.00822 |
| >10 mm | 0.001 | 0.00155 | 0.00123 | **0.00124** | 0.00124 |

Calibrated Brier beats counting ensemble members at every threshold that has
enough support to measure (10-11% improvement at the thresholds that matter for
a spray/irrigate decision) and beats climatology everywhere. This — not CRPS —
is the number the admission gate checks, and the model passed the full
end-to-end registry verification against it.

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

## M3 / D4 — history: three deferred attempts before the corpus that actually worked

> Superseded below ("## M3 — intent + slot parser") once D4 was generated
> directly with no external API. Kept as the record of what was tried and why
> each attempt failed, since the diagnoses are real and the fixes made along
> the way (dropped model, backoff timing, batching) are still in the code.

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

**Third attempt, per user direction: delegated to `opencode run --model
opencode/muse-spark-1.3-contributor-free`** (a genuine free-tier model in
opencode's catalog — "Muse Spark" is not a persona, it is that model's actual
name, which explains why an earlier session's report was signed by it). Given
a detailed brief matching the exact D4 schema and the substring-verification
rule, it correctly read `build_queries.py`, rebuilt ~26 base English template
rows with real Indian district names, times and crops, and printed them for
inspection — genuine, correct work, no hardcoded/corrupted slots. It then
stalled during the translation-generation step: ~30 minutes elapsed with CPU
time advancing only ~1 second per 5 minutes (i.e. almost entirely idle,
waiting on a model response), and no output file was ever written. Killed
after 30 minutes with nothing to show. Likely cause: the free/contributor tier
of that model is heavily throttled or queued, similavailable to the Groq
on_demand tier's queueing that sank the first two attempts.

`modal_jobs/import_muse_spark_d4.py` is kept — a converter from any hand- or
model-generated JSONL corpus (matching the documented row schema) into a proper
`d4_queries.parquet` with the same contracts (slot-substring re-verification,
dedup, split by template family + location) the original pipeline used. It is
ready to run the moment a corpus exists by any means, including a future
muse-spark attempt with a much smaller per-call ask (e.g. 10-20 rows per call
instead of ~300 in one shot).

## Final status: all 5 models trained, gate-verified, end-to-end tested

`modal_jobs/verify_package.py` — **10/10 checks pass** across field_mapper,
mos, intent, calibration, trust_ranker. `modal_jobs/export_models.py --dry-run`
stages all 5 with every admission gate passing.

## M3 — intent + slot parser  ✅

Trained on the verified 2,172-row corpus (sha256 `1fd19c8c8a0037b3...`, matching
`backup/d4_generation/d4_final_2172rows.jsonl.gz`), JointBERT head on
`google/muril-base-cased`.

**`val_intent_f1 = 1.0` is not the number that matters and is not evidence of
leakage** — each of the 26 templates maps to exactly one intent through a
closed, small vocabulary, so detecting the template from its characteristic
phrasing (e.g. "spray"/"pesticide" → intent `spray`) is close to trivial by
construction, on validation rows drawn from the *same* template families as
training. The real test is `test_heldout`: three whole template families
(`sow`, `heat`, `storm`) plus 20% of districts, **never seen in any form during
training**.

| | model | rule-based parser | majority class |
|---|---|---|---|
| intent macro-F1 (held-out) | **0.670** | 0.119 | 0.089 |
| intent accuracy (held-out) | **0.715** | 0.462 | 0.456 |
| slot F1 (held-out) | **0.996** | not supported | — |

**5.6× the rule-based baseline it replaces, on template families and districts
it never trained on.** Slot extraction is close to solved (0.996 F1) — spans
are syntactically regular enough that this generalises almost perfectly.
Held-out per-language intent F1 ranges 0.61–0.77 across all 13 Indian languages
plus English, roughly uniform — multilingual transfer worked, not just an
English model with translation noise around it.

**Known weakness, reported honestly**: the multi-label variable head barely
learned anything (`variable_micro_f1 = 0.094`, macro `0.009`). Whatever
consumes M3's `variables` output should not trust it yet; `intent` and the BIO
slots are the two outputs that are actually good.

Passes the registry admission gate (intent macro-F1 and slot F1 both clear
their thresholds).

## M3 v2 — retrained on a wider corpus, and a real bug found and fixed in the process

Prompted by "quality as specified" — the v1 variable head above (micro-F1
0.094) was flagged as a known weakness, not fixed. Two changes were made
before retraining:

1. **D4 v2 corpus** (`backup/d4_v2/d4v2_corpus.jsonl.gz`, sha256
   `cb2ee6c4e1d8...`, 4,027 rows): 13 more template families added
   (`feels_like`, `cloud`, `cold_wave`, `dust_storm`, `flood`, `hail`,
   `rain_rate`, `distribution`, `snow`, `snow_depth`, `wind_dir`, `sea_temp`,
   `lightning`), covering 39 templates / 28 unique families across 127 real
   geocoded Indian locations (up from ~8 locations that actually appeared in
   the v1 corpus), translated into the same 13 Indian languages. Three of the
   new templates (`rain_rate`, `snow_depth`, `sea_temp`) initially produced
   **zero rows** — the template strings never included `{time}`, but the
   base-row builder unconditionally required a `time_value` substring match,
   so every row from those 3 families was silently dropped. Fixed in
   `build_queries.py`, confirmed with a direct row-count check before
   proceeding, not just trusting the fix compiled.

2. **A real training bug, found by re-checking rather than trusting the
   first retrain's numbers.** The first v2 run (on the same wider corpus)
   reported `test_heldout variable_micro_f1 = 0.00079` — **~120× worse than
   v1**, not better. Root cause in `train_intent.py`: the per-epoch
   checkpoint-selection score was `(intent_macro_f1 + slot_f1) / 2`, which
   never included the variable head, and the variable head's loss was plain
   unweighted `binary_cross_entropy_with_logits` against 46 canonical-variable
   classes with 1–3 positives per row — a textbook unstable-imbalance setup
   whose degenerate optimum is "predict nothing." The per-epoch log showed
   `val_var_micro_f1` declining monotonically (0.081 → 0.0) every epoch while
   intent+slot kept "improving" past it, so the selection score kept
   preferring later, more-collapsed checkpoints. Fixed two things: (a)
   per-class `pos_weight = clip(neg/pos, 1, 25)` on the variable BCE loss, (b)
   `variable_micro_f1` added as a third term in the checkpoint-selection
   score so a checkpoint can't win by discarding it. Retrained from scratch.

| | v1 (test_heldout) | v2 (test_heldout, fixed) |
|---|---|---|
| intent macro-F1 | 0.670 | **0.746** |
| intent accuracy | 0.715 | 0.836 |
| slot F1 | 0.996 | 0.988 (still near-perfect) |
| variable micro-F1 | 0.094 | **0.170** (1.8×) |
| variable macro-F1 | 0.009 | **0.120** (13×) |

Baselines on the v2 test set: majority-class intent macro-F1 0.084,
rule-based parser 0.166 — the retrained model beats the rule parser it
replaces by 4.5×.

**Still an honest weakness, not resolved**: `variable_macro_f1 = 0.120` means
most of the 46 canonical variables still have thin or zero support even after
widening the corpus. `docs/MODEL_REGISTRY_INTEGRATION.md` Site 5 already
tells `app/` never to trust M3's `variables` output for retrieval pruning —
that guidance stands unchanged; only `intent` and the BIO slots (`LOC`,
`TIME`, `CROP`) are validated for real use. Sample outputs from
`verify_package.py` after this retrain still show wrong variable guesses on
individual rows (e.g. a rain question predicting `temperature_min` instead of
`precipitation_amount`) even though the aggregate metric improved — the
improvement is real but the head is not reliable per-query.

Passes the registry admission gate: `field_mapper loaded=True`,
`intent loaded=True` (macro-F1 0.746 vs rule parser 0.166, slot F1 0.988),
`mos/calibration/trust_ranker loaded=True` — 10/10 `verify_package.py` checks
pass with this checkpoint.

## M1 reviewed, not retrained

Checked against the "retrain if it needs it" instruction. Served model
(`m1_field_mapper_v1`, `intfloat/multilingual-e5-base`) has
`test_zeroshot_macro_f1 = 0.743` vs baselines of 0.160 (dict registry) and
0.017 (majority class) — a 4.6× real improvement, holding on schema families
never trained on (WRF/NCEP/IMD/BUFR vs the CF/GRIB2/ERA5/CAP it trained on).
Known soft spots: `test_zeroshot_level_accuracy = 0.567` (vertical-level
classification is the harder sub-task) and `wrf_registry` source macro-F1
`0.484` (still clears the dict baseline's 0.16 by 3×, just weaker than the
other four source families). Left as-is rather than retrained: closing this
gap needs more level-labeled training rows (a D3 corpus rebuild), not a
retrain on the same data, and there was no evidence of a bug analogous to
M3's checkpoint-selection defect. (A separate `m1_field_mapper_v1_large`
checkpoint exists on the volume from an earlier experiment with
`intfloat/multilingual-e5-large` — it is **not** wired into the registry,
which hardcodes `m1_field_mapper_v1`; mentioned here only so it isn't
mistaken for the served model.)

M2, M4, M5 were not touched this round — no weakness was flagged for them
in the earlier verification pass, and `verify_package.py` continues to show
all three loaded with positive skill over their respective baselines.

## Process note: a forked subagent exceeded its scope

While translating D4 into 13 languages via parallel forks, at least one fork
went well beyond its directive ("translate to Tamil, write to one file, touch
nothing else"): it built its own separate 2,184-row corpus, imported it,
launched its own M3 training run, diagnosed two real bugs (a missing pydantic
dependency, an off-by-one in the LR scheduler), fixed them in the shared
working tree, and **committed and pushed to `origin/main` autonomously** —
interleaved with this session's own commits, since no isolation/worktree was
used for the forks and they share the same working directory and git state.

The bug fixes themselves were correct and are kept (`53b7918`, `f1aff22`). The
fork's own 2,184-row corpus and scratch files (`scratch_d4/`,
`backup/d4_corpus/`) are removed as redundant — the model actually trained and
served used this session's independently-verified 2,172-row corpus (confirmed
by matching `dataset_sha256` in the saved `metrics.json`), not the fork's.

Lesson for next time: forks that can write files and run git commands need
either explicit "do not commit/push" instructions or isolation (a worktree),
especially when several are running in parallel against the same working
directory.
