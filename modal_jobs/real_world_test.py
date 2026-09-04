"""Real-world smoke test: run the trained models on realistic, hand-picked
inputs (not training data) and print what a caller would actually see."""
from __future__ import annotations

from modal_jobs.common import MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, timeout=300)
def run():
    from weathergpt_models import ModelRegistry

    registry = ModelRegistry.from_dir(MODEL_DIR)
    print("=== registry status ===")
    for row in registry.status():
        print(f"  {row['model']:14s} loaded={row['loaded']}  {row['reason']}")

    print("\n=== M1: field mapper on fields it was NEVER trained to map ===")
    real_fields = [
        # (name, unit, description, level, time_range, evidence_hint)
        ("2t", "K", "", "surface", "", None),
        ("sp", "Pa", "Surface pressure", "surface", "", None),
        ("Cyclonic Storm", None, "cyclone warning for coastal district", "", "", "warning"),
        ("swh", "m", "Significant height of combined wind waves and swell", "", "", None),
        ("SNOWH", "mm", "Physical snow depth", "", "", None),  # unit contradiction on purpose
        ("d2m", "K", "2 metre dewpoint temperature", "2 m above ground", "", None),
    ]
    for name, unit, desc, level, tr, hint in real_fields:
        m = registry.field_mapper.map_field(name, unit=unit, description=desc,
                                            level_text=level, time_range_text=tr,
                                            evidence_class_hint=hint)
        result = "ABSTAIN" if not m.is_usable else m.canonical_variable
        print(f"  {name:16s} unit={str(unit):8s} -> {result:26s} "
              f"({m.source}, conf={m.confidence:.2f})")

    print("\n=== M2: MOS correction for a real Sept monsoon scenario (Mumbai-ish) ===")
    scenarios = [
        ("temperature_2m", {"gfs_seamless": 29.8, "ecmwf_ifs025": 28.9,
                            "icon_seamless": 29.3, "gem_seamless": 30.1},
         {"lead_hours": 18, "lead_age_days": 0, "hour_utc": 12, "doy": 250,
          "elevation_m": 8.0, "lat": 19.07, "lon": 72.87}),
        ("precipitation", {"gfs_seamless": 12.5, "ecmwf_ifs025": 4.2,
                           "icon_seamless": 18.0, "gem_seamless": 2.0},
         {"lead_hours": 30, "lead_age_days": 1, "hour_utc": 15, "doy": 250,
          "elevation_m": 8.0, "lat": 19.07, "lon": 72.87}),
        ("wind_speed_10m", {"gfs_seamless": 22.0, "ecmwf_ifs025": 27.5,
                            "icon_seamless": 24.0, "gem_seamless": 30.0},
         {"lead_hours": 6, "lead_age_days": 0, "hour_utc": 6, "doy": 250,
          "elevation_m": 8.0, "lat": 19.07, "lon": 72.87}),
    ]
    for variable, forecasts, context in scenarios:
        others = {v: f for v, f, _ in scenarios if v != variable}
        c = registry.mos.correct(variable, forecasts=forecasts, context=context,
                                 other_forecasts=others)
        print(f"  {variable:16s} raw_mean={c.raw_ensemble_mean:7.2f} -> corrected={c.value:7.2f} "
              f"(Δ{c.correction:+.2f})  80% interval [{c.interval_low:.2f}, {c.interval_high:.2f}]")

    print("\n=== M4: calibrated rain probability for the same monsoon scenario ===")
    precip_forecasts = scenarios[1][1]
    precip_context = scenarios[1][2]
    for threshold in (0.1, 1.0, 5.0, 10.0, 25.0):
        p = registry.calibration.exceedance_probability(
            "precipitation", threshold, forecasts=precip_forecasts, context=precip_context)
        print(f"  P(rain > {threshold:5.1f} mm) = {p.probability:.3f}  "
              f"(raw member frequency would have said {p.raw_ensemble_frequency:.3f})")

    print("\n=== M5: which source to trust, right now, for this scenario ===")
    for variable, forecasts, context in scenarios:
        ranked = registry.trust_ranker.rank(variable, candidates=forecasts,
                                            context={**context, "month": 9})
        order = " > ".join(f"{r.source.split('_')[0]}({r.value:.1f})" for r in ranked)
        print(f"  {variable:16s} {order}")


@app.local_entrypoint()
def main_real_world_test():
    run.remote()
