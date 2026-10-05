"""Train the per-target event models (gradient boosting; one model per target).

Three kinds of target, each with its own honest evaluation:

  binary    event happened / did not (thunderstorm, fog, strong wind, hot day ...).
            LightGBM logloss -> isotonic calibration on the validation period.
            Reported: Brier skill vs climatology, AUC, PR-AUC, ECE, and CSI/POD/FAR
            against a single-feature physical rule tuned on the same validation data.
  xent      fractional label in [0,1]: the share of a 0.5 deg cell that saw >= X mm
            (= probability that a random point in the cell sees it).  Cross-entropy
            objective accepts fractional labels directly.
  quantile  a RANGE: q10 / q50 / q90.  One model per quantile, optional cube-root
            space for rain.  Reported: pinball loss, interval coverage and width
            vs a static-width interval around the raw GFS value, and median MAE vs
            raw GFS.  Intervals are widened by a conformal margin fitted on the
            validation period; coverage is then re-measured on the test sets.

Every metric is reported on three sets: `val` (used for early stopping and
calibration), `test_time` (same places, future dates) and `test_space` (places the
model never saw, future dates).  Nothing is tuned on test.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from event_models import rain_products as rp

SETS = ("val", "test_time", "test_space")


@dataclass
class Target:
    name: str
    table: str                      # 'step' | 'day' | 'rain'
    kind: str                       # 'binary' | 'xent' | 'quantile' | 'curve'
    label: str
    primary: str | None = None      # physical feature for the baseline rule (binary/xent)
    direction: int = 1              # +1: larger feature -> event, -1: smaller feature -> event
    point_forecast: str | None = None   # raw GFS column the range is compared against (quantile)
    quantiles: tuple = (0.1, 0.5, 0.9)
    transform: str = "identity"     # 'cbrt' for rain amounts
    max_rows: int = 3_000_000
    notes: str = ""
    # 'curve' targets: one label column per threshold, in increasing threshold order
    labels: tuple | None = None
    thresholds: tuple | None = None
    score_low: str | None = None        # raw GFS score used for thresholds < 15 mm (baseline)
    score_high: str | None = None       # raw GFS score used for thresholds >= 15 mm (baseline)
    key_thresholds: tuple = (1.0, 2.5, 15.6, 64.5)
    wet: float = 1.0                    # 'it rained' level for the conditional-amount score of a curve


# ----------------------------------------------------------------------------- metrics
def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    y, score = np.asarray(y)[ok], np.asarray(score)[ok]
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def _pr_auc(y, score):
    from sklearn.metrics import average_precision_score
    ok = np.isfinite(score)
    y, score = np.asarray(y)[ok], np.asarray(score)[ok]
    return float(average_precision_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def ece(p, y, bins=10):
    order = np.argsort(p)
    p, y = np.asarray(p)[order], np.asarray(y)[order]
    chunks = np.array_split(np.arange(len(p)), bins)
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)) / len(p))


def contingency(pred, obs):
    tp = int((pred & obs).sum())
    fn = int((~pred & obs).sum())
    fp = int((pred & ~obs).sum())
    return {"pod": tp / max(tp + fn, 1), "far": fp / max(tp + fp, 1), "csi": tp / max(tp + fn + fp, 1),
            "hits": tp, "misses": fn, "false_alarms": fp}


def best_threshold(score, obs, grid=200):
    """Threshold on `score` maximising CSI, searched over score percentiles."""
    qs = np.unique(np.nanpercentile(score, np.linspace(50, 99.9, grid)))
    best_t, best_csi = qs[-1], -1.0
    for t in qs:
        csi = contingency(score >= t, obs)["csi"]
        if csi > best_csi:
            best_t, best_csi = float(t), csi
    return best_t, best_csi


def breakdown(frame, p, y, by, min_n=500, min_pos=10):
    out = {}
    base = float(np.nanmean(y))
    for key, idx in frame.groupby(by, observed=True).indices.items():
        if len(idx) < min_n or (y[idx] >= 0.5).sum() < min_pos:
            continue
        yy, pp = y[idx], p[idx]
        brier = float(np.mean((pp - yy) ** 2))
        clim = float(np.mean((yy.mean() - yy) ** 2))
        out[str(key)] = {"n": int(len(idx)), "rate": round(float(yy.mean()), 4),
                         "auc": round(_auc(yy >= 0.5, pp), 3),
                         "bss_vs_local_clim": round(1 - brier / clim, 3) if clim > 0 else None}
    return out


def climatology_refs(frame, y, split):
    """Reference probabilities built from TRAIN rows only.

    global      one base rate -- flattering, because season and region alone beat it by a mile
    zone_month  climate zone x calendar month -- the fair reference for places never trained on
    point_month the place's own monthly rate -- only defined for places that were in training
    """
    tr = (split == "train") & np.isfinite(y)
    glob = float(np.mean(y[tr]))
    refs = {"global": np.full(len(frame), glob)}
    if "zone" in frame and "month" in frame:
        keys = pd.DataFrame({"zone": frame["zone"].to_numpy(), "month": frame["month"].to_numpy(), "y": y})
        zm = keys[tr].groupby(["zone", "month"])["y"].mean().rename("ref")
        refs["zone_month"] = keys.join(zm, on=["zone", "month"])["ref"].fillna(glob).to_numpy()
    if "point_id" in frame and "month" in frame:
        keys = pd.DataFrame({"point_id": frame["point_id"].to_numpy(), "month": frame["month"].to_numpy(), "y": y})
        pm = keys[tr].groupby(["point_id", "month"])["y"].mean().rename("ref")
        refs["point_month"] = keys.join(pm, on=["point_id", "month"])["ref"].to_numpy()   # NaN: unseen place
    return refs


def bss(pred, y, ref, min_coverage=0.95):
    ok = np.isfinite(ref)
    if ok.mean() < min_coverage:
        return None
    brier, clim = np.mean((pred[ok] - y[ok]) ** 2), np.mean((ref[ok] - y[ok]) ** 2)
    return round(float(1 - brier / clim), 4) if clim > 0 else None


def binary_metrics(name, frame, p_cal, y, rule_score, threshold_ml, threshold_rule, refs):
    obs = y >= 0.5
    brier = float(np.mean((p_cal - y) ** 2))
    clim = float(np.mean((refs["global"] - y) ** 2))
    m = {"n": int(len(y)), "base_rate": round(float(y.mean()), 5),
         "brier": round(brier, 6), "brier_clim": round(clim, 6),
         "bss_vs_global": bss(p_cal, y, refs["global"]),
         "bss_vs_zone_month": bss(p_cal, y, refs["zone_month"]) if "zone_month" in refs else None,
         "bss_vs_point_month": bss(p_cal, y, refs["point_month"]) if "point_month" in refs else None,
         "auc": round(_auc(obs, p_cal), 4), "pr_auc": round(_pr_auc(obs, p_cal), 4),
         "auc_rule_feature": round(_auc(obs, rule_score), 4), "ece": round(ece(p_cal, y), 5),
         "ml": {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in contingency(p_cal >= threshold_ml, obs).items()},
                "threshold": round(float(threshold_ml), 5)},
         "rule": {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in contingency(rule_score >= threshold_rule, obs).items()},
                  "threshold": round(float(threshold_rule), 4)}}
    m["bss_vs_train_clim"] = m["bss_vs_global"]      # kept for older readers; the honest headline is zone_month
    return m


def calibrated_interval(q_lo, q_mid, q_hi, margin):
    """The served interval: [q_lo - margin, q_hi + margin], kept around the median.

    The conformal margin is NEGATIVE when validation over-covers (it narrows the range).  A large negative margin could
    otherwise invert the interval or leave its own median outside it, so the interval is clamped to contain q_mid.
    Used by BOTH the evaluation here and the serving model, so reported coverage is the coverage that is served.
    """
    return np.minimum(q_lo - margin, q_mid), np.maximum(q_hi + margin, q_mid)


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


# ----------------------------------------------------------------------------- training
def _fwd(y, transform):
    return np.cbrt(y) if transform == "cbrt" else y


def _inv(y, transform):
    return np.clip(y, 0, None) ** 3 if transform == "cbrt" else y


def _subsample(idx, y, max_rows, rng, positives_first):
    if len(idx) <= max_rows:
        return idx
    if positives_first:
        pos = idx[y[idx] >= 0.5]
        neg = idx[y[idx] < 0.5]
        keep_neg = rng.choice(neg, size=min(len(neg), max(max_rows - len(pos), max_rows // 2)), replace=False)
        return np.concatenate([pos, keep_neg])
    return rng.choice(idx, size=max_rows, replace=False)


def _fit(lgb, params, X_tr, y_tr, X_va, y_va, rounds, time_limit_s, label):
    start = time.time()

    def time_guard(env):
        if time.time() - start > time_limit_s:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    dval = lgb.Dataset(X_va, y_va, reference=dtrain, free_raw_data=True)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200), time_guard])
    print(f"[train] {label}: {booster.best_iteration or booster.current_iteration()} rounds, "
          f"{time.time() - start:.0f}s", flush=True)
    return booster


BASE_PARAMS = dict(learning_rate=0.06, num_leaves=127, min_data_in_leaf=400, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, verbose=-1,
                   num_threads=0, seed=7)


def fit_target(target: Target, frame: pd.DataFrame, features: list, split: np.ndarray, out_dir: Path,
               *, rounds: int = 1500, time_limit_s: int = 2400, extra_params: dict | None = None,
               min_rows: int = 1000) -> dict:
    if target.kind == "curve":
        return fit_curve(target, frame, features, split, out_dir, rounds=rounds, time_limit_s=time_limit_s,
                         extra_params=extra_params, min_rows=min_rows)
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(7)
    X = frame[features].to_numpy("float32")
    y = frame[target.label].to_numpy("float64")
    ok = np.isfinite(y)
    idx = {s: np.where((split == s) & ok)[0] for s in ("train", *SETS)}
    if min(len(idx["train"]), len(idx["val"])) < min_rows:
        return {"skipped": f"too few rows (train {len(idx['train'])}, val {len(idx['val'])})"}
    out_dir.mkdir(parents=True, exist_ok=True)
    zones = frame["zone"].to_numpy() if "zone" in frame else None
    report = {"target": target.name, "kind": target.kind, "table": target.table, "label": target.label, "n_features": len(features),
              "rows": {s: int(len(i)) for s, i in idx.items()}, "notes": target.notes}

    if target.kind in ("binary", "xent"):
        objective = "binary" if target.kind == "binary" else "cross_entropy"
        train_idx = _subsample(idx["train"], y, target.max_rows, rng, positives_first=True)
        report["rows"]["train_used"] = int(len(train_idx))
        report["train_rate_full"] = float(y[idx["train"]].mean())
        params = {**BASE_PARAMS, "objective": objective, "metric": "cross_entropy" if objective != "binary" else "binary_logloss",
                  **(extra_params or {})}
        booster = _fit(lgb, params, X[train_idx], y[train_idx], X[idx["val"]], y[idx["val"]], rounds, time_limit_s, target.name)
        raw = {s: booster.predict(X[idx[s]], num_iteration=booster.best_iteration or None) for s in ("val", *SETS[1:])}
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw["val"], y[idx["val"]])
        cal = {s: iso.predict(raw[s]) for s in raw}
        rule_all = target.direction * frame[target.primary].to_numpy("float64")
        obs_val = y[idx["val"]] >= 0.5
        thr_ml, _ = best_threshold(cal["val"], obs_val)
        thr_rule, _ = best_threshold(rule_all[idx["val"]], obs_val)
        report["metrics"] = {}
        refs_all = climatology_refs(frame, y, split)
        for s in SETS:
            ii = idx[s]
            m = binary_metrics(target.name, frame.iloc[ii], cal[s], y[ii], rule_all[ii], thr_ml, thr_rule,
                               {k: v[ii] for k, v in refs_all.items()})
            if s != "val":
                sub = frame.iloc[ii].copy()
                sub["season"] = np.where(sub["month"].isin([6, 7, 8, 9]), "monsoon_JJAS", "other")
                m["by_zone"] = breakdown(sub, cal[s], y[ii], "zone")
                m["by_season"] = breakdown(sub, cal[s], y[ii], "season")
                lead_col = "lead_bucket" if "lead_bucket" in sub else None
                if lead_col:
                    m["by_lead"] = breakdown(sub, cal[s], y[ii], lead_col)
            report["metrics"][s] = m
        report["reliability_test_time"] = _reliability(cal["test_time"], y[idx["test_time"]])
        booster.save_model(str(out_dir / "model.txt"), num_iteration=booster.best_iteration or None)
        (out_dir / "calibrator.json").write_text(json.dumps(
            {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
             "threshold_ml": thr_ml, "threshold_rule": thr_rule}))
        report["top_features"] = _top_features(booster, features)

    else:  # quantile
        taus = target.quantiles
        train_idx = _subsample(idx["train"], y, target.max_rows, rng, positives_first=False)
        report["rows"]["train_used"] = int(len(train_idx))
        yt = _fwd(y, target.transform)
        preds = {s: [] for s in ("val", *SETS[1:])}
        for tau in taus:
            params = {**BASE_PARAMS, "objective": "quantile", "alpha": tau, "metric": "quantile", **(extra_params or {})}
            booster = _fit(lgb, params, X[train_idx], yt[train_idx], X[idx["val"]], yt[idx["val"]], rounds, time_limit_s, f"{target.name} q{tau}")
            for s in preds:
                preds[s].append(booster.predict(X[idx[s]], num_iteration=booster.best_iteration or None))
            booster.save_model(str(out_dir / f"model_q{int(round(tau * 100)):02d}.txt"), num_iteration=booster.best_iteration or None)
        q = {s: _inv(np.sort(np.stack(v, axis=1), axis=1), target.transform) for s, v in preds.items()}  # monotone quantiles
        lo_i, hi_i = 0, len(taus) - 1
        nominal = taus[hi_i] - taus[lo_i]
        # conformalised widening: margin so the validation interval reaches nominal coverage
        yv = y[idx["val"]]
        scores = np.maximum(q["val"][:, lo_i] - yv, yv - q["val"][:, hi_i])
        margin = float(np.quantile(scores, min(1.0, nominal * (1 + 1 / len(scores)))))
        raw_fc = frame[target.point_forecast].to_numpy("float64") if target.point_forecast else None
        resid = None
        if raw_fc is not None:
            tr = idx["train"]
            resid = y[tr] - raw_fc[tr]
            r_lo, r_hi = np.nanquantile(resid, taus[lo_i]), np.nanquantile(resid, taus[hi_i])
        report["interval"] = {"nominal": round(nominal, 3), "conformal_margin": round(margin, 4)}
        report["metrics"] = {}
        for s in SETS:
            ii, yy, qq = idx[s], y[idx[s]], q[s]
            lo_c, hi_c = calibrated_interval(qq[:, lo_i], qq[:, len(taus) // 2], qq[:, hi_i], margin)
            m = {"n": int(len(ii)),
                 "pinball": {f"q{int(round(t * 100)):02d}": round(pinball(yy, qq[:, k], t), 5) for k, t in enumerate(taus)},
                 "coverage_raw": round(float(((yy >= qq[:, lo_i]) & (yy <= qq[:, hi_i])).mean()), 4),
                 "coverage_conformal": round(float(((yy >= lo_c) & (yy <= hi_c)).mean()), 4),
                 "mean_width_raw": round(float((qq[:, hi_i] - qq[:, lo_i]).mean()), 4),
                 "mean_width_conformal": round(float((hi_c - lo_c).mean()), 4),
                 "median_mae": round(float(np.mean(np.abs(yy - qq[:, len(taus) // 2]))), 4)}
            static_in = None
            if raw_fc is not None:
                f = raw_fc[ii]
                good = np.isfinite(f)
                m["gfs_raw_mae"] = round(float(np.mean(np.abs(yy[good] - f[good]))), 4)
                m["gfs_raw_bias"] = round(float(np.mean(f[good] - yy[good])), 4)
                m["median_bias"] = round(float(np.mean(qq[:, len(taus) // 2][good] - yy[good])), 4)
                static_lo, static_hi = f + r_lo, f + r_hi
                static_in = np.where(good, (yy >= static_lo) & (yy <= static_hi), np.nan)
                m["static_interval_coverage"] = round(float(np.nanmean(static_in)), 4)
                m["static_interval_width"] = round(float((static_hi - static_lo)[good].mean()), 4)
            if zones is not None and s != "val":
                zz = zones[ii]
                inside = (yy >= lo_c) & (yy <= hi_c)
                by_zone = {}
                for z in np.unique(zz):
                    sel = zz == z
                    if sel.sum() < 500:
                        continue
                    entry = {"n": int(sel.sum()), "coverage": round(float(inside[sel].mean()), 3)}
                    if static_in is not None:   # the uniform-coverage advantage is the point of a range model
                        entry["static_coverage"] = round(float(np.nanmean(static_in[sel])), 3)
                    by_zone[str(z)] = entry
                m["coverage_by_zone"] = by_zone
            report["metrics"][s] = m
        (out_dir / "interval.json").write_text(json.dumps({"quantiles": list(taus), "transform": target.transform,
                                                          "conformal_margin": margin}))
    (out_dir / "features.json").write_text(json.dumps(features))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def _reliability(p, y, bins=10):
    order = np.argsort(p)
    out = []
    for c in np.array_split(order, bins):
        out.append({"mean_p": round(float(p[c].mean()), 4), "obs_rate": round(float(y[c].mean()), 4), "n": int(len(c))})
    return out


def _top_features(booster, names, k=12):
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[:k]
    total = gain.sum() or 1.0
    return [{"feature": names[i], "gain_share": round(float(gain[i] / total), 4)} for i in order]


# ----------------------------------------------------------------------------- exceedance curve
def _tier_sample(rng, F, T, budget_rows):
    """Importance-sample training rows so rare heavy-rain rows survive; weights = 1/p keep estimates unbiased.

    Tiers (by what the cell reached): heavy (>=25 mm somewhere) / wet (>=2.5 mm) / dry.  A uniform
    subsample of a few million rows would contain almost no 64 mm cells.
    """
    i25, i25m = int(np.argmin(np.abs(T - 25.0))), int(np.argmin(np.abs(T - 2.5)))
    heavy = F[:, i25] > 0
    wet = (F[:, i25m] > 0) & ~heavy
    rel = np.where(heavy, 12.0, np.where(wet, 4.0, 1.0))
    if len(F) <= budget_rows:
        return np.arange(len(F)), np.ones(len(F), "float32")
    lo, hi = 0.0, 1.0
    for _ in range(60):                       # bisection on the scale so that sum(p) = budget
        mid = (lo + hi) / 2
        if np.minimum(1.0, mid * rel).sum() > budget_rows:
            hi = mid
        else:
            lo = mid
    p = np.minimum(1.0, lo * rel)
    keep = rng.random(len(F)) < p
    return np.nonzero(keep)[0], (1.0 / p[keep]).astype("float32")


def _stack(Xb, log_t):
    """(M, F) rows x (T,) thresholds -> (M*T, F+1): every row repeated per threshold with log(t) appended."""
    m, n_t = len(Xb), len(log_t)
    out = np.empty((m * n_t, Xb.shape[1] + 1), "float32")
    out[:, :-1] = np.repeat(Xb, n_t, axis=0)
    out[:, -1] = np.tile(log_t, m).astype("float32")
    return out


def _predict_curve(booster, Xb, log_t, chunk=400_000):
    iteration = booster.best_iteration or None
    out = np.empty((len(Xb), len(log_t)), "float32")
    for j, lt in enumerate(log_t):
        for a in range(0, len(Xb), chunk):
            part = Xb[a:a + chunk]
            col = np.full((len(part), 1), lt, "float32")
            out[a:a + chunk, j] = booster.predict(np.concatenate([part, col], axis=1), num_iteration=iteration)
    return out


def _group_means(F, keys: list, mask):
    """Mean label per group of `keys` (list of arrays) over `mask` rows -> DataFrame indexed by the keys."""
    frame = pd.DataFrame(F[mask])
    for i, k in enumerate(keys):
        frame[f"_k{i}"] = np.asarray(k)[mask]
    return frame.groupby([f"_k{i}" for i in range(len(keys))]).mean()


def _lookup(table, keys, fallback):
    got = table.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy()
    return np.where(np.isfinite(got), got, fallback[None, :]) if fallback is not None else got


def fit_curve(target: Target, frame: pd.DataFrame, features: list, split: np.ndarray, out_dir: Path, *,
              rounds: int = 1200, time_limit_s: int = 3600, extra_params: dict | None = None,
              eval_cap: int = 500_000, max_stacked: int = 6_000_000, min_rows: int = 1000) -> dict:
    """One model for the whole rain-exceedance curve S(t) = P(share of cell >= t), t in `thresholds`.

    The threshold enters as log(t) with a monotone-DEcreasing constraint, so P(>=25mm) can never exceed
    P(>=2.5mm) for any input, and rare heavy classes borrow strength from the common light ones.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(7)
    T = np.asarray(target.thresholds, float)
    n_t = len(T)
    log_t = np.log(T)
    F = frame[list(target.labels)].to_numpy("float32")
    X = frame[features].to_numpy("float32")
    ok = np.isfinite(F).all(axis=1)
    idx = {s: np.nonzero((split == s) & ok)[0] for s in ("train", *SETS)}
    if min(len(idx["train"]), len(idx["val"])) < min_rows:
        return {"skipped": f"too few rows (train {len(idx['train'])}, val {len(idx['val'])})"}
    out_dir.mkdir(parents=True, exist_ok=True)
    zone = frame["zone"].to_numpy() if "zone" in frame else None
    month = frame["month"].to_numpy() if "month" in frame else None
    point = frame["point_id"].to_numpy() if "point_id" in frame else None
    report = {"target": target.name, "kind": "curve", "table": target.table, "thresholds": list(map(float, T)), "n_features": len(features),
              "wet": target.wet, "rows": {s: int(len(i)) for s, i in idx.items()}, "notes": target.notes}

    # ---- train on an importance-sampled stack of (row, threshold) pairs
    tr = idx["train"]
    pick, w = _tier_sample(rng, F[tr], T, max_stacked // n_t)
    tr_sel = tr[pick]
    report["rows"]["train_base_used"] = int(len(tr_sel))
    Xs, ys, ws = _stack(X[tr_sel], log_t), F[tr_sel].reshape(-1), np.repeat(w, n_t)
    va_base = rng.choice(idx["val"], size=min(80_000, len(idx["val"])), replace=False)   # uniform: unbiased early stopping
    Xv, yv = _stack(X[va_base], log_t), F[va_base].reshape(-1)
    constraints = [0] * len(features) + [-1]
    params = {**BASE_PARAMS, "objective": "cross_entropy", "metric": "cross_entropy", "min_data_in_leaf": 800,
              "monotone_constraints": constraints, "monotone_constraints_method": "basic", **(extra_params or {})}
    start = time.time()

    def time_guard(env):
        if time.time() - start > time_limit_s:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    booster = lgb.train(params, lgb.Dataset(Xs, ys, weight=ws, free_raw_data=True), num_boost_round=rounds,
                        valid_sets=[lgb.Dataset(Xv, yv, free_raw_data=True)],
                        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100), time_guard])
    print(f"[train] {target.name}: {booster.best_iteration or booster.current_iteration()} rounds, "
          f"{time.time() - start:.0f}s on {len(Xs):,} stacked rows", flush=True)
    del Xs, ys, ws, Xv, yv

    # ---- calibrate per threshold on a uniform validation sample; floor on event counts
    cal_rows = rng.choice(idx["val"], size=min(300_000, len(idx["val"])), replace=False)
    P_val = _predict_curve(booster, X[cal_rows], log_t)
    isos, cal_info = [], []
    for j in range(n_t):
        y_j = F[cal_rows, j]
        n_events = int((y_j > 0).sum())
        if n_events >= 300:
            isos.append(IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(P_val[:, j], y_j))
        else:
            isos.append(None)           # too few validation events: keep the monotone model's own output
        cal_info.append({"threshold": float(T[j]), "val_events": n_events, "calibrated": isos[j] is not None})
    report["calibration"] = cal_info

    def calibrate(P):
        out = np.stack([isos[j].predict(P[:, j]) if isos[j] is not None else P[:, j] for j in range(P.shape[1])], axis=1)
        return rp.enforce_monotone(out)

    # ---- references from TRAIN rows only
    glob = F[tr].mean(axis=0)
    zm_table = _group_means(F, [zone, month], np.isin(np.arange(len(F)), tr)) if zone is not None and month is not None else None
    pm_table = _group_means(F, [point, month], np.isin(np.arange(len(F)), tr)) if point is not None and month is not None else None
    cal_val_scores = {}
    if target.score_low or target.score_high:
        for j in range(n_t):
            col = target.score_high if (T[j] >= 15.0 and target.score_high) else target.score_low
            cal_val_scores[j] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                frame[col].to_numpy("float64")[cal_rows], F[cal_rows, j])

    key_idx = [int(np.argmin(np.abs(T - k))) for k in target.key_thresholds]
    report["metrics"] = {}
    for s in SETS:
        ii_all = idx[s]
        ii = rng.choice(ii_all, size=min(eval_cap, len(ii_all)), replace=False)
        P_raw = _predict_curve(booster, X[ii], log_t)
        P = calibrate(P_raw)
        Y = F[ii]
        m = {"n_rows_total": int(len(ii_all)), "n_evaluated": int(len(ii)),
             "monotone_violations_raw": round(float((np.diff(P_raw, axis=1) > 1e-6).any(axis=1).mean()), 6),
             "monotone_violations_calibrated": round(float((np.diff(P, axis=1) > 1e-9).any(axis=1).mean()), 6),
             "thresholds": {}}
        zm_ref = _lookup(zm_table, [zone[ii], month[ii]], glob) if zm_table is not None else None
        pm_ref = _lookup(pm_table, [point[ii], month[ii]], None) if pm_table is not None else None
        for j in range(n_t):
            y_j, p_j = Y[:, j], P[:, j]
            brier = float(np.mean((p_j - y_j) ** 2))
            entry = {"threshold_mm": float(T[j]), "base_rate": round(float(y_j.mean()), 6), "brier": round(brier, 7),
                     "bss_vs_global": bss(p_j, y_j, np.full(len(y_j), glob[j])),
                     "bss_vs_zone_month": bss(p_j, y_j, zm_ref[:, j]) if zm_ref is not None else None,
                     "bss_vs_point_month": bss(p_j, y_j, pm_ref[:, j]) if pm_ref is not None else None}
            if j in cal_val_scores:
                col = target.score_high if (T[j] >= 15.0 and target.score_high) else target.score_low
                gfs_p = cal_val_scores[j].predict(frame[col].to_numpy("float64")[ii])
                entry["bss_vs_gfs_calibrated"] = bss(p_j, y_j, gfs_p)     # beats a one-variable calibrated GFS?
                raw_score = frame[col].to_numpy("float64")[ii]
            else:
                raw_score = None
            obs = y_j >= 0.5
            if obs.sum() >= 50:
                entry["auc"] = round(_auc(obs, p_j), 4)
                if raw_score is not None:
                    entry["auc_raw_gfs"] = round(_auc(obs, raw_score), 4)
            entry["events_ge_half_cell"] = int(obs.sum())
            # rare classes: "half the cell reached it" almost never happens, so also score discrimination of
            # "at least one pixel of the cell reached it" using the predicted share as the ranking score
            anyp = y_j > 0
            entry["events_any_pixel"] = int(anyp.sum())
            if anyp.sum() >= 50:
                entry["auc_any_pixel"] = round(_auc(anyp, p_j), 4)
                if raw_score is not None:
                    entry["auc_any_pixel_raw_gfs"] = round(_auc(anyp, raw_score), 4)
            m["thresholds"][f"{T[j]:g}"] = entry
        if s != "val":
            detail = {}
            for j in key_idx:
                y_j, p_j = Y[:, j], P[:, j]
                d = {"reliability": _reliability(p_j, y_j)} if s == "test_time" else {}
                if zone is not None:
                    zz = zone[ii]
                    d["by_zone"] = {str(z): {"n": int((zz == z).sum()),
                                             "bss_vs_zone_month": bss(p_j[zz == z], y_j[zz == z], zm_ref[zz == z, j])}
                                    for z in np.unique(zz) if (zz == z).sum() >= 2000 and zm_ref is not None}
                if "lead_bucket" in frame:
                    lb = frame["lead_bucket"].to_numpy()[ii]
                    d["by_lead"] = {str(l): {"n": int((lb == l).sum()),
                                             "bss_vs_zone_month": bss(p_j[lb == l], y_j[lb == l], zm_ref[lb == l, j])}
                                    for l in np.unique(lb) if (lb == l).sum() >= 2000 and zm_ref is not None}
                detail[f"{T[j]:g}"] = d
            m["detail_key_thresholds"] = detail
            # "if yes, how much": conditional-median amount vs the pixel-level truth, where it clearly rained
            true_med, _ = rp.amount_quantiles(Y, T, levels=(0.5,), wet=target.wet)
            pred_med, capped = rp.amount_quantiles(P, T, levels=(0.5,), wet=target.wet)
            wet_rows = (rp.survival_at(Y, T, target.wet) >= 0.2) & np.isfinite(true_med[:, 0]) & np.isfinite(pred_med[:, 0])
            if wet_rows.sum() > 200:
                err = np.log10(pred_med[wet_rows, 0] / true_med[wet_rows, 0])
                amount = {"rows": int(wet_rows.sum()), "log10_mae": round(float(np.mean(np.abs(err))), 4),
                          "log10_bias": round(float(np.mean(err)), 4),
                          "within_factor_2": round(float((np.abs(err) <= np.log10(2)).mean()), 4)}
                if (target.score_low or "rain_cell_mean_mm") in frame:
                    raw_col = target.score_low or "rain_cell_mean_mm"
                    gfs_amt = np.maximum(frame[raw_col].to_numpy("float64")[ii][wet_rows], target.wet)
                    gerr = np.log10(gfs_amt / true_med[wet_rows, 0])
                    amount["gfs_raw_log10_mae"] = round(float(np.mean(np.abs(gerr))), 4)
                    amount["gfs_raw_within_factor_2"] = round(float((np.abs(gerr) <= np.log10(2)).mean()), 4)
                m["amount_if_wet_median"] = amount
        report["metrics"][s] = m
    booster.save_model(str(out_dir / "model.txt"), num_iteration=booster.best_iteration or None)
    (out_dir / "calibrators.json").write_text(json.dumps({
        "thresholds": list(map(float, T)),
        "calibrators": [None if iso is None else {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()} for iso in isos]}))
    clim = {"global": glob.tolist(), "thresholds": list(map(float, T))}
    if zm_table is not None:
        clim["zone_month"] = {f"{z}|{mo}": row.tolist() for (z, mo), row in zip(zm_table.index, zm_table.to_numpy())}
    (out_dir / "climatology.json").write_text(json.dumps(clim))
    (out_dir / "features.json").write_text(json.dumps(features))
    report["top_features"] = _top_features(booster, features + ["log_threshold"])
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    return report
