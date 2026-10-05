"""Event labels and continuous truth derived from real METAR observations.

Input is the IEM ASOS/METAR CSV (`network=IN__ASOS`): `valid` (UTC), `tmpf`,
`dwpf`, `sknt`, `gust` (knots), `vsby` (statute miles), `wxcodes` (present
weather, space separated, e.g. "-TSRA BR", "VCTS", "FZFG").

ERA5 has no thunderstorm, fog or visibility fields (CAPE and visibility come
back 100% null from the Open-Meteo archive -- verified), so these labels are
the only honest ground truth for those targets.  They are observations at an
airport, which is what the models therefore predict: "will the station see X",
not "is X happening somewhere in this 25 km cell".

Two time bases, deliberately:
  * hourly labels use the UTC hour the observation falls in;
  * daily temperature extremes use the IST calendar day, because that is the
    day a user asking "will it be hot tomorrow" means, and Tmin (about 00:30
    UTC) would otherwise land on a UTC day boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MILE_KM = 1.609344
KT_MS = 0.514444
IST = pd.Timedelta(hours=5, minutes=30)

# Thresholds are physical / IMD-style, fixed here so every model and every
# evaluation uses the same definition of the event.
FOG_VIS_KM = 1.0          # WMO definition of fog
DENSE_FOG_VIS_KM = 0.2
STRONG_WIND_KT = 25       # ~46 km/h peak gust-or-mean
GALE_KT = 34              # ~63 km/h
HOT_DAY_C = 40.0          # IMD heat-wave Tmax floor for plains
SEVERE_HEAT_C = 45.0
COLD_NIGHT_C = 5.0


def _tokens(wx: str) -> list:
    return [t.lstrip("+-") for t in str(wx).split()] if isinstance(wx, str) else []


def parse_wxcodes(series: pd.Series) -> pd.DataFrame:
    """Present-weather string -> boolean columns, one row per observation.

    'VC' tokens (in the vicinity, 8-16 km) are kept separate from at-station
    weather: VCTS is a thunderstorm near the airport, TS is over it.
    """
    tokens = series.map(_tokens)
    raw = series.fillna("").astype(str)

    def any_token(pred):
        return tokens.map(lambda ts: any(pred(t) for t in ts))

    at_station = lambda t: not t.startswith("VC")  # noqa: E731
    out = pd.DataFrame(index=series.index)
    out["ts_at"] = any_token(lambda t: at_station(t) and "TS" in t)
    out["ts_vc"] = any_token(lambda t: t.startswith("VC") and "TS" in t)
    out["ts_any"] = out["ts_at"] | out["ts_vc"]
    out["rain_any"] = any_token(lambda t: at_station(t) and ("RA" in t or "DZ" in t))
    out["rain_heavy"] = raw.map(lambda s: any(tok.startswith("+") and "RA" in tok
                                              for tok in s.split()))
    out["dust"] = any_token(lambda t: at_station(t) and any(c in t for c in ("DU", "SA", "SS", "DS", "PO")))
    out["squall"] = any_token(lambda t: "SQ" in t)
    out["fog_code"] = any_token(lambda t: at_station(t) and t in ("FG", "FZFG"))
    return out


def rh_from_dewpoint(temp_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """Magnus formula; clipped because METAR rounding can put dew a tenth above temp."""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * dew_c / (b + dew_c)) / np.exp(a * temp_c / (b + temp_c))
    return rh.clip(0, 100)


def clean_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw IEM rows -> typed observations with physical sanity bounds applied.

    Values outside plausible Indian surface ranges become NaN rather than being
    dropped, so a bad sensor reading cannot delete a thunderstorm report that
    shares the row.
    """
    obs = pd.DataFrame({"valid": pd.to_datetime(frame["valid"], utc=True)})
    for column in ("tmpf", "dwpf", "sknt", "gust", "vsby"):
        obs[column] = pd.to_numeric(frame.get(column), errors="coerce")
    obs["temp_c"] = (obs["tmpf"] - 32.0) * 5.0 / 9.0
    obs["dew_c"] = (obs["dwpf"] - 32.0) * 5.0 / 9.0
    obs.loc[~obs["temp_c"].between(-30, 55), "temp_c"] = np.nan
    obs.loc[~obs["dew_c"].between(-40, 40), "dew_c"] = np.nan
    obs["rh"] = rh_from_dewpoint(obs["temp_c"], obs["dew_c"])
    obs.loc[~obs["sknt"].between(0, 150), "sknt"] = np.nan
    obs.loc[~obs["gust"].between(0, 200), "gust"] = np.nan
    obs["vis_km"] = obs["vsby"] * MILE_KM
    obs.loc[~obs["vis_km"].between(0, 100), "vis_km"] = np.nan
    obs["peak_kt"] = obs[["sknt", "gust"]].max(axis=1, skipna=True)
    wx = frame["wxcodes"] if "wxcodes" in frame else pd.Series("", index=frame.index)
    obs = pd.concat([obs, parse_wxcodes(wx)], axis=1)
    obs["wxcodes"] = wx.fillna("").astype(str)  # raw string kept so a later pass can re-parse
    return obs.sort_values("valid").drop_duplicates("valid").reset_index(drop=True)


def hourly_labels(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (UTC hour) with at least one observation.

    Event flags are OR-ed over the hour's reports (an hour that contained a
    thunderstorm report counts as a thunderstorm hour).  Continuous truth is the
    hour's mean (min for visibility, max for peak wind).  `n_obs` is kept so a
    later cleaning pass can drop hours that are one stray report.
    """
    hour = obs["valid"].dt.floor("h")
    grouped = obs.groupby(hour)
    out = pd.DataFrame({
        "n_obs": grouped.size(),
        "temp_c": grouped["temp_c"].mean(),
        "rh": grouped["rh"].mean(),
        "wind_ms": grouped["sknt"].mean() * KT_MS,
        "peak_wind_ms": grouped["peak_kt"].max() * KT_MS,
        "vis_km": grouped["vis_km"].min(),
        "peak_kt": grouped["peak_kt"].max(),
        "ts_at": grouped["ts_at"].max(),
        "ts_any": grouped["ts_any"].max(),
        "rain_any": grouped["rain_any"].max(),
        "rain_heavy": grouped["rain_heavy"].max(),
        "dust": grouped["dust"].max(),
        "squall": grouped["squall"].max(),
    })
    out["fog"] = out["vis_km"] < FOG_VIS_KM
    out["dense_fog"] = out["vis_km"] < DENSE_FOG_VIS_KM
    out["strong_wind"] = out["peak_kt"] >= STRONG_WIND_KT
    out["gale"] = out["peak_kt"] >= GALE_KT
    out.index.name = "hour_utc"
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype(bool)
    return out.reset_index()


def daily_temperature_labels(obs: pd.DataFrame, min_obs: int = 8) -> pd.DataFrame:
    """IST-day Tmax/Tmin and the hot-day / cold-night flags.

    A day with fewer than `min_obs` temperature reports is dropped, not
    labelled: a Tmax from three night-time reports is not a Tmax.
    """
    ist_day = (obs["valid"] + IST).dt.floor("D").dt.tz_localize(None)
    valid = obs.dropna(subset=["temp_c"])
    grouped = valid.groupby(ist_day.loc[valid.index])["temp_c"]
    out = pd.DataFrame({"tmax_c": grouped.max(), "tmin_c": grouped.min(), "n_temp": grouped.size()})
    out = out[out["n_temp"] >= min_obs].copy()
    out["hot_day"] = out["tmax_c"] >= HOT_DAY_C
    out["severe_heat"] = out["tmax_c"] >= SEVERE_HEAT_C
    out["cold_night"] = out["tmin_c"] <= COLD_NIGHT_C
    out.index.name = "ist_day"
    return out.reset_index()
