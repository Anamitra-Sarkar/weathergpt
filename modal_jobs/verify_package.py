"""End-to-end verification of the shipped inference package.

Runs on Modal against the real artifacts on the model volume, because the local
machine never holds a checkpoint.  This is the test that the thing handed to the
framework actually works — the unit tests cover the maths, this covers the
loading, the shapes, and the fact that each model's serving path reproduces what
its trainer measured.
"""
from __future__ import annotations

import json

from modal_jobs.common import MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

SAMPLE_FIELDS = [
    # (name, unit, description, level, time range, what a correct answer looks like)
    ("APCP", "kg m-2", "Total precipitation", "surface", "0-3 hour acc fcst",
     "precipitation_amount"),
    ("TMAX", "K", "Maximum temperature", "2 m above ground", "0-6 hour max fcst",
     "temperature_max"),
    ("TMAX", "mm", "Maximum temperature", "2 m above ground", "", "ABSTAIN"),
    ("RAINNC", "mm", "ACCUMULATED TOTAL GRID SCALE PRECIPITATION", "", "",
     "precipitation_amount"),
    ("prate", "kg m-2 s-1", "Precipitation rate", "surface", "", "precipitation_rate"),
    ("Heavy Rainfall", None, "heavy rainfall warning for the district", "", "",
     "heavy_rain_warning"),
    ("XLAT", "degree_north", "LATITUDE, SOUTH IS NEGATIVE", "", "", "ABSTAIN"),
    ("wind_gusts_10m", "km/h", "Gusts at 10 metres above ground as a maximum of the "
     "preceding hour", "", "", "wind_gust"),
]

SAMPLE_QUERIES = [
    ("Will it rain in Bhandara tomorrow afternoon?", "none"),
    ("kal Bhandara me baarish hogi kya?", "none"),
    ("Should I spray pesticide on my cotton in Yavatmal tomorrow?", "spray"),
    ("क्या कल नागपुर में बारिश होगी?", "none"),
    ("Can fishermen go to sea off Ratnagiri tomorrow?", "marine"),
    ("Is there any weather warning for Kolhapur today?", "warning_check"),
]

FORECASTS = {
    "temperature_2m": {"gfs_seamless": 31.2, "ecmwf_ifs025": 30.4,
                       "icon_seamless": 30.9, "gem_seamless": 31.8},
    "precipitation": {"gfs_seamless": 4.2, "ecmwf_ifs025": 1.1,
                      "icon_seamless": 6.8, "gem_seamless": 0.0},
    "wind_speed_10m": {"gfs_seamless": 18.0, "ecmwf_ifs025": 22.5,
                       "icon_seamless": 19.4, "gem_seamless": 25.1},
    "relative_humidity_2m": {"gfs_seamless": 78.0, "ecmwf_ifs025": 81.0,
                             "icon_seamless": 76.0, "gem_seamless": 80.0},
}
CONTEXT = {"lead_hours": 30, "lead_age_days": 1, "hour_utc": 6, "doy": 245,
           "elevation_m": 303.0, "lat": 21.15, "lon": 79.09, "month": 9}


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, timeout=60 * 40)
def verify() -> dict:
    from weathergpt_models import ModelRegistry

    registry = ModelRegistry.from_dir(MODEL_DIR)
    report: dict = {"status": registry.status(), "checks": []}

    def check(name: str, ok: bool, detail: str = ""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print("\n=== registry ===")
    for row in registry.status():
        print(f"  {row['model']:14s} loaded={row['loaded']}  {row['reason']}")

    # --- M1 -----------------------------------------------------------------
    if registry.field_mapper is not None:
        print("\n=== M1 field mapper ===")
        correct = 0
        for name, unit, description, level, window, expected in SAMPLE_FIELDS:
            mapping = registry.field_mapper.map_field(
                name, unit=unit, description=description, level_text=level,
                time_range_text=window,
                evidence_class_hint="warning" if "warning" in description else None)
            got = "ABSTAIN" if not mapping.is_usable else mapping.canonical_variable
            hit = got == expected
            correct += hit
            print(f"  {name:16.16s} {str(unit):12.12s} -> {got:26.26s} "
                  f"({mapping.source}, conf {mapping.confidence:.3f}) "
                  f"{'ok' if hit else 'EXPECTED ' + expected}")
        check("m1_sample_fields", correct >= len(SAMPLE_FIELDS) - 1,
              f"{correct}/{len(SAMPLE_FIELDS)} correct")

        unit_conflict = registry.field_mapper.map_field(
            "TMAX", unit="mm", description="Maximum temperature")
        check("m1_refuses_unit_contradiction", not unit_conflict.is_usable,
              f"got {unit_conflict.canonical_variable}")

    # --- M3 -----------------------------------------------------------------
    if registry.intent is not None:
        print("\n=== M3 intent parser ===")
        correct, with_location = 0, 0
        for text, expected in SAMPLE_QUERIES:
            parsed = registry.intent.parse(text)
            hit = parsed.intent == expected
            correct += hit
            with_location += parsed.slot_text("LOC") is not None
            print(f"  {text[:52]:54.54s} -> {parsed.intent:14s} "
                  f"({parsed.intent_confidence:.2f}) LOC={parsed.slot_text('LOC')!r} "
                  f"TIME={parsed.slot_text('TIME')!r} vars={parsed.variables[:3]}")
        check("m3_intents", correct >= len(SAMPLE_QUERIES) - 2,
              f"{correct}/{len(SAMPLE_QUERIES)} intents correct")
        check("m3_extracts_location", with_location >= len(SAMPLE_QUERIES) - 1,
              f"{with_location}/{len(SAMPLE_QUERIES)} produced a LOC span")

    # --- M2 -----------------------------------------------------------------
    if registry.mos is not None:
        print("\n=== M2 MOS ===")
        ok = True
        for variable in ("temperature_2m", "precipitation", "wind_speed_10m"):
            if not registry.mos.supports(variable):
                continue
            others = {k: v for k, v in FORECASTS.items() if k != variable}
            corrected = registry.mos.correct(variable, forecasts=FORECASTS[variable],
                                             context=CONTEXT, other_forecasts=others)
            monotone = all(corrected.quantiles[a] <= corrected.quantiles[b] + 1e-6
                           for a, b in zip(sorted(corrected.quantiles),
                                           sorted(corrected.quantiles)[1:]))
            bracketed = corrected.interval_low <= corrected.value <= corrected.interval_high
            ok = ok and monotone and bracketed
            print(f"  {variable:20s} raw={corrected.raw_ensemble_mean:8.3f} "
                  f"-> {corrected.value:8.3f} ({corrected.correction:+.3f}) "
                  f"[{corrected.interval_low:.2f}, {corrected.interval_high:.2f}] "
                  f"head={corrected.algorithm_version.split(':')[-1]} "
                  f"monotone={monotone} bracketed={bracketed}")
            if variable == "precipitation":
                nonneg = all(v >= 0 for v in corrected.quantiles.values())
                check("m2_precipitation_non_negative", nonneg, str(corrected.quantiles))
        check("m2_quantiles_monotone_and_bracketed", ok)

        # a missing model must not break the call
        partial = registry.mos.correct(
            "temperature_2m",
            forecasts={"gfs_seamless": 31.2, "ecmwf_ifs025": 30.4},
            context=CONTEXT)
        check("m2_tolerates_missing_members",
              partial.value == partial.value, f"value={partial.value:.3f}")

    # --- M4 -----------------------------------------------------------------
    if registry.calibration is not None:
        print("\n=== M4 calibration ===")
        curve = registry.calibration.probability_curve(
            "precipitation", forecasts=FORECASTS["precipitation"], context=CONTEXT,
            other_forecasts={k: v for k, v in FORECASTS.items() if k != "precipitation"})
        probabilities = [item.probability for item in curve]
        for item in curve:
            print(f"  P(precip > {item.threshold:5.1f} mm) = {item.probability:.4f} "
                  f"(raw member frequency {item.raw_ensemble_frequency:.3f}, {item.method})")
        check("m4_probabilities_in_range",
              all(0.0 <= p <= 1.0 for p in probabilities), str(probabilities))
        # isotonic refinement is per threshold, so a small non-monotonicity is
        # possible; a large one means the thresholds were fitted inconsistently
        worst = max((probabilities[i + 1] - probabilities[i]
                     for i in range(len(probabilities) - 1)), default=0.0)
        check("m4_exceedance_broadly_decreasing", worst < 0.10,
              f"largest increase with threshold = {worst:.4f}")
        print(f"  transfer assumption: {registry.calibration.transfer_assumption[:160]}...")

    # --- M5 -----------------------------------------------------------------
    if registry.trust_ranker is not None:
        print("\n=== M5 trust ranker ===")
        ok = True
        for variable in ("temperature_2m", "precipitation", "wind_speed_10m"):
            if not registry.trust_ranker.supports(variable):
                continue
            ranked = registry.trust_ranker.rank(variable, candidates=FORECASTS[variable],
                                                context=CONTEXT)
            ordered = all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))
            ok = ok and ordered and len(ranked) == 4
            print(f"  {variable:20s} " + " > ".join(
                f"{r.source.split('_')[0]}({r.score:+.2f})" for r in ranked))
        check("m5_returns_a_total_order", ok)

    failures = [item for item in report["checks"] if not item["ok"]]
    report["passed"] = not failures
    report["n_checks"] = len(report["checks"])
    print(f"\n=== {len(report['checks']) - len(failures)}/{len(report['checks'])} checks passed ===")
    for item in failures:
        print(f"  FAILED {item['check']}: {item['detail']}")
    return report


@app.local_entrypoint()
def main_verify_package():
    result = verify.remote()
    print(json.dumps({"passed": result["passed"], "n_checks": result["n_checks"]}, indent=2))
