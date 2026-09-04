"""M5 — learned evidence-trust ranker.

`app/services/ranker.py` orders evidence with a fixed hand-tuned formula,
`0.4*authority + 0.25*freshness + 0.20*spatial + 0.15*quality`, where the
authority term is a static table.  For a single query point at a single valid
time the freshness, spatial and quality terms are identical across candidate
NWP sources, so the formula collapses to **one global preference order that
never changes** — the same model wins in Leh in January and in Kochi in July.

That is testable, and it is wrong.  Model skill in India is strongly
conditional: ECMWF tends to win on temperature at long lead, the higher
resolution models win in orographic terrain, and the monsoon changes the
precipitation ordering entirely.

This trains a LambdaMART ranker over candidate sources per (location, valid
time, lead) group, with relevance derived from which source was actually
closest to the ERA5 truth.  It is scored on ranking quality (NDCG, pairwise
accuracy) and — the number that matters — the error you incur by *following* it,
against the fixed authority order it replaces.

Historical-skill features are computed on the training window only and joined
by (location, source, variable, lead bucket), so no future information reaches
a training row.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from modal_jobs.common import DATA_DIR, MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

ALGORITHM_VERSION = "m5_trust_ranker_v1"
TARGETS = ("temperature_2m", "precipitation", "wind_speed_10m")

# The static order implied by app/services/ranker.py's AUTHORITY table for the
# four NWP sources we actually hold, highest authority first.
HEURISTIC_AUTHORITY_ORDER = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless")


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, timeout=60 * 120, memory=32768, cpu=8)
def train(seed: int = 42, max_groups: int = 400_000, num_boost_round: int = 400) -> dict:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from modal_jobs.features_d1 import MODELS, dataset_sha256, load_d1, split_masks

    rng = np.random.default_rng(seed)
    frame, files = load_d1(DATA_DIR)
    masks = split_masks(frame, seed=seed)
    print(f"[m5] {len(frame):,} rows, {frame['loc_id'].nunique()} locations, "
          f"cutoff={masks['cutoff']}")

    results: dict = {}
    boosters: dict = {}

    for target in TARGETS:
        forecast_columns = [f"fc_{target}_{model}" for model in MODELS]
        forecasts = frame[forecast_columns].to_numpy(dtype="float64")
        truth = frame[f"truth_{target}"].to_numpy(dtype="float64")

        # A group is only rankable when every candidate produced a value; a
        # partial group would make relevance depend on who was missing.
        complete = np.isfinite(forecasts).all(1) & np.isfinite(truth)
        print(f"[m5:{target}] {complete.sum():,} complete groups of {len(MODELS)} candidates")

        errors = np.abs(forecasts - truth[:, None])

        # Relevance: best source gets len(MODELS)-1, worst gets 0.  Ties broken
        # by the fixed authority order so a genuinely tied group does not teach
        # the model an arbitrary preference.
        authority_rank = np.array([HEURISTIC_AUTHORITY_ORDER.index(model) for model in MODELS])
        order = np.lexsort((authority_rank[None, :].repeat(len(errors), 0), errors), axis=1)
        relevance = np.empty_like(order)
        np.put_along_axis(relevance, order,
                          np.arange(len(MODELS) - 1, -1, -1)[None, :].repeat(len(errors), 0), axis=1)

        # --- historical per-source skill, from the TRAINING window only ------
        train_rows = masks["train"] & complete
        lead_bucket = (frame["lead_age_days"].to_numpy() // 2).astype(int)
        skill_frame = pd.DataFrame({
            "loc_id": np.repeat(frame["loc_id"].to_numpy()[train_rows], len(MODELS)),
            "lead_bucket": np.repeat(lead_bucket[train_rows], len(MODELS)),
            "source": np.tile(np.array(MODELS), int(train_rows.sum())),
            "abs_error": errors[train_rows].reshape(-1),
        })
        by_loc = (skill_frame.groupby(["loc_id", "source"])["abs_error"].mean()
                  .rename("hist_mae_loc").reset_index())
        by_loc_lead = (skill_frame.groupby(["loc_id", "source", "lead_bucket"])["abs_error"]
                       .mean().rename("hist_mae_loc_lead").reset_index())
        global_mae = skill_frame.groupby("source")["abs_error"].mean().to_dict()

        loc_lookup = {(row.loc_id, row.source): row.hist_mae_loc
                      for row in by_loc.itertuples()}
        loc_lead_lookup = {(row.loc_id, row.source, row.lead_bucket): row.hist_mae_loc_lead
                           for row in by_loc_lead.itertuples()}

        index = np.flatnonzero(complete)
        # Subsample groups for tractability; ranking needs whole groups, not rows.
        split_of = np.full(len(frame), "", dtype=object)
        for split_name in ("train", "val", "test"):
            split_of[masks[split_name]] = split_name

        chosen = {}
        for split_name in ("train", "val", "test"):
            candidates = index[split_of[index] == split_name]
            cap = max_groups if split_name == "train" else max_groups // 4
            if len(candidates) > cap:
                candidates = rng.choice(candidates, cap, replace=False)
            chosen[split_name] = np.sort(candidates)
            print(f"[m5:{target}] {split_name}: {len(candidates):,} groups")
        if min(len(v) for v in chosen.values()) < 2000:
            raise RuntimeError(f"{target}: a split has too few complete groups to rank")

        source_onehot = np.eye(len(MODELS))
        loc_ids = frame["loc_id"].to_numpy()
        elevation = frame["elevation_m"].to_numpy(dtype="float64")
        latitude = frame["lat"].to_numpy(dtype="float64")
        longitude = frame["lon"].to_numpy(dtype="float64")
        lead_hours = frame["lead_hours"].to_numpy(dtype="float64")
        lead_age = frame["lead_age_days"].to_numpy(dtype="float64")
        month = frame["month"].to_numpy(dtype="float64")
        doy = frame["doy"].to_numpy(dtype="float64")

        feature_names = ([f"is_{model}" for model in MODELS] +
                         ["forecast", "deviation_from_mmm", "ensemble_sd", "ensemble_range",
                          "hist_mae_loc", "hist_mae_loc_lead", "hist_mae_global",
                          "authority_prior", "lead_hours", "lead_age_days",
                          "sin_doy", "cos_doy", "elevation_km", "lat", "lon", "is_monsoon"])

        def build(rows: np.ndarray):
            block_forecast = forecasts[rows]
            multi_model_mean = block_forecast.mean(1, keepdims=True)
            spread = block_forecast.std(1, keepdims=True)
            span = (block_forecast.max(1) - block_forecast.min(1))[:, None]
            n = len(rows)
            features = np.empty((n * len(MODELS), len(feature_names)))
            for position, model in enumerate(MODELS):
                slice_index = slice(position, None, len(MODELS))
                block = np.concatenate([
                    np.tile(source_onehot[position], (n, 1)),
                    block_forecast[:, position:position + 1],
                    block_forecast[:, position:position + 1] - multi_model_mean,
                    spread, span,
                    np.array([[loc_lookup.get((loc_ids[r], model), global_mae[model])]
                              for r in rows]),
                    np.array([[loc_lead_lookup.get(
                        (loc_ids[r], model, int(lead_bucket[r])),
                        loc_lookup.get((loc_ids[r], model), global_mae[model]))] for r in rows]),
                    np.full((n, 1), global_mae[model]),
                    np.full((n, 1), float(len(MODELS) - HEURISTIC_AUTHORITY_ORDER.index(model))),
                    lead_hours[rows][:, None], lead_age[rows][:, None],
                    np.sin(2 * np.pi * doy[rows] / 365.25)[:, None],
                    np.cos(2 * np.pi * doy[rows] / 365.25)[:, None],
                    (elevation[rows] / 1000.0)[:, None],
                    latitude[rows][:, None], longitude[rows][:, None],
                    np.isin(month[rows], [6, 7, 8, 9]).astype(float)[:, None],
                ], axis=1)
                features[slice_index] = block
            labels = relevance[rows].reshape(-1)
            groups = np.full(n, len(MODELS))
            return features, labels, groups

        Xtr, ytr, gtr = build(chosen["train"])
        Xva, yva, gva = build(chosen["val"])
        Xte, yte, gte = build(chosen["test"])

        booster = lgb.train(
            {"objective": "lambdarank", "metric": "ndcg",
             "ndcg_eval_at": [1, 2, 3], "lambdarank_truncation_level": len(MODELS),
             "learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 100,
             "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 1,
             "verbose": -1, "seed": seed, "label_gain": list(range(len(MODELS) + 1))},
            lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feature_names),
            num_boost_round=num_boost_round,
            valid_sets=[lgb.Dataset(Xva, label=yva, group=gva, reference=None)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])

        def ndcg_at(scores, labels, k: int) -> float:
            scores = scores.reshape(-1, len(MODELS))
            labels = labels.reshape(-1, len(MODELS))
            ranked = np.argsort(-scores, axis=1)[:, :k]
            gains = np.take_along_axis(labels, ranked, axis=1)
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = (gains * discounts).sum(1)
            ideal = np.sort(labels, axis=1)[:, ::-1][:, :k]
            idcg = (ideal * discounts).sum(1)
            return float(np.mean(np.where(idcg > 0, dcg / np.maximum(idcg, 1e-9), 0.0)))

        def follow_error(rows: np.ndarray, picks: np.ndarray) -> float:
            """RMSE incurred by taking the chosen source's value as the answer."""
            values = forecasts[rows][np.arange(len(rows)), picks]
            return float(np.sqrt(np.mean((values - truth[rows]) ** 2)))

        report = {}
        for split_name, X, y, rows in (("val", Xva, yva, chosen["val"]),
                                        ("test_spatial_holdout", Xte, yte, chosen["test"])):
            scores = booster.predict(X)
            model_picks = scores.reshape(-1, len(MODELS)).argmax(1)
            best_possible = np.abs(forecasts[rows] - truth[rows][:, None]).argmin(1)
            heuristic_pick = MODELS.index(HEURISTIC_AUTHORITY_ORDER[0])

            pairwise = 0.0
            scores_2d = scores.reshape(-1, len(MODELS))
            labels_2d = y.reshape(-1, len(MODELS))
            concordant = total = 0
            for a in range(len(MODELS)):
                for b in range(a + 1, len(MODELS)):
                    different = labels_2d[:, a] != labels_2d[:, b]
                    if not different.any():
                        continue
                    agree = ((labels_2d[different, a] > labels_2d[different, b]) ==
                             (scores_2d[different, a] > scores_2d[different, b]))
                    concordant += int(agree.sum())
                    total += int(different.sum())
            pairwise = concordant / max(total, 1)

            report[split_name] = {
                "n_groups": int(len(rows)),
                "ndcg@1": ndcg_at(scores, y, 1),
                "ndcg@2": ndcg_at(scores, y, 2),
                "ndcg@3": ndcg_at(scores, y, 3),
                "pairwise_accuracy": pairwise,
                "top1_is_actually_best_rate": float(np.mean(model_picks == best_possible)),
                "heuristic_top1_is_actually_best_rate": float(
                    np.mean(best_possible == heuristic_pick)),
                "random_top1_is_actually_best_rate": 1.0 / len(MODELS),
                "rmse_following_ranker": follow_error(rows, model_picks),
                "rmse_following_fixed_authority": follow_error(
                    rows, np.full(len(rows), heuristic_pick)),
                "rmse_multi_model_mean": float(np.sqrt(np.mean(
                    (forecasts[rows].mean(1) - truth[rows]) ** 2))),
                "rmse_oracle_best_source": follow_error(rows, best_possible),
            }
            reference = report[split_name]["rmse_following_fixed_authority"]
            report[split_name]["rmse_skill_vs_fixed_authority"] = (
                1 - report[split_name]["rmse_following_ranker"] / reference)

        # which source the ranker actually prefers, and how that varies —
        # the direct evidence that a fixed order is the wrong model
        test_scores = booster.predict(Xte).reshape(-1, len(MODELS))
        picks = test_scores.argmax(1)
        preference = {MODELS[i]: float(np.mean(picks == i)) for i in range(len(MODELS))}
        by_lead = {}
        test_lead = lead_age[chosen["test"]]
        for value in sorted(set(test_lead.tolist())):
            mask = test_lead == value
            if mask.sum() < 200:
                continue
            by_lead[str(int(value))] = {MODELS[i]: float(np.mean(picks[mask] == i))
                                        for i in range(len(MODELS))}

        importance = dict(sorted(zip(feature_names,
                                     booster.feature_importance("gain").tolist()),
                                 key=lambda item: -item[1])[:12])
        results[target] = {**report, "source_preference_test": preference,
                           "global_source_mae": {model: float(global_mae[model])
                                                 for model in MODELS},
                           "source_preference_by_lead_age": by_lead,
                           "feature_importance_gain_top12": importance,
                           "best_iteration": int(booster.best_iteration or num_boost_round)}
        boosters[target] = booster
        print(f"[m5:{target}] TEST ndcg@1={report['test_spatial_holdout']['ndcg@1']:.4f} "
              f"top1-correct {report['test_spatial_holdout']['top1_is_actually_best_rate']:.3f} "
              f"vs fixed {report['test_spatial_holdout']['heuristic_top1_is_actually_best_rate']:.3f} | "
              f"rmse {report['test_spatial_holdout']['rmse_following_ranker']:.3f} "
              f"vs {report['test_spatial_holdout']['rmse_following_fixed_authority']:.3f} "
              f"(oracle {report['test_spatial_holdout']['rmse_oracle_best_source']:.3f})")

    out_dir = f"{MODEL_DIR}/{ALGORITHM_VERSION}"
    os.makedirs(out_dir, exist_ok=True)
    for target, booster in boosters.items():
        booster.save_model(f"{out_dir}/lambdamart_{target}.txt")

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "model_kind": "LightGBM LambdaMART over candidate NWP sources per (location, "
                      "valid time, lead) group",
        "dataset_kind": "d1_multi_model_nwp_vs_era5_seamless",
        "dataset_shards": len(files), "dataset_rows": int(len(frame)),
        "dataset_sha256": dataset_sha256(files),
        "split": "chronological 70% cutoff AND 20% spatially held-out locations",
        "split_cutoff": masks["cutoff"], "held_out_locations": masks["held_out_locations"],
        "candidate_sources": list(MODELS),
        # served as the fallback per-source skill for a location with no history
        "global_source_skill": {model: float(np.mean([
            results[target]["global_source_mae"][model] for target in TARGETS]))
            for model in MODELS},
        "heuristic_baseline": "fixed authority order from app/services/ranker.py "
                              f"({' > '.join(HEURISTIC_AUTHORITY_ORDER)})",
        "seed": seed, "trained_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    with open(f"{out_dir}/metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    from modal_jobs.common import MODEL_VOL
    MODEL_VOL.commit()
    print(json.dumps(metrics, indent=2, default=str)[:6000])
    return metrics


@app.local_entrypoint()
def main_train_trust_ranker():
    train.remote()
