"""Build, for ONE query location, exactly the feature tables the models were trained on.

There is deliberately no second implementation of any feature.  Each method below runs the same collector functions
(`event_models.collect_gfs`, `collect_gfs_ext`) on the live run's grids and then the same table builders and
feature functions (`event_models.dataset`, `features_events`) that `run_training.build_tables` uses, in the same
order.  The parity tests compare these frames to the training tables row for row.

Reference climatology: the query is referenced to its nearest grid NODE, which is exactly the reference error training
simulated for stations (see `features_events.nearest_node_ids`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from event_models import collect_gfs as base
from event_models import collect_gfs_ext as ext
from event_models import dataset
from event_models import features_events as fe
from weathergpt_events import static as static_mod

QUERY_ID = "Q:query"
KEYS1 = ["point_id", "run_date", "lead_h"]
KEYS2 = ["point_id", "run_date", "day_k"]


@dataclass
class RunArrays:
    """One GFS/GEFS run on the India window, as the collectors produce it."""
    run_date: date
    base: dict      # {lead: {field: 2-D array}}   (collect_gfs.fetch_run)
    ext: dict       # {lead: {field: 2-D array}}   (collect_gfs_ext.fetch_ext_run)


class FeatureBuilder:
    def __init__(self, grids: dict, climatology: pd.DataFrame, nodes: pd.DataFrame):
        """`climatology` is the trainer's `climatology_day` table (point_id, bin, <col>...); `nodes` the points table."""
        self.grids = grids
        self.clim = climatology.set_index(["point_id", "bin"]) if "bin" in climatology.columns else climatology
        self.nodes = nodes
        # Coastal places were part of training (METAR stations whose 0.25 deg cell is partly sea), so "land-only" means "no less land than the
        # training stations had" (their 5th percentile), never stricter than the 0.5 the grid nodes satisfy.
        stations = nodes[nodes["kind"] == "station"] if {"kind", "land_frac"} <= set(nodes.columns) else nodes.iloc[0:0]
        self.min_land = min(0.5, float(stations["land_frac"].quantile(0.05))) if len(stations) >= 20 else 0.5
        self.terrain = {"dzdx": grids["dzdx"], "dzdy": grids["dzdy"]}

    # ------------------------------------------------------------------ location
    def locate(self, lat: float, lon: float, point_id: str | None = None, static: pd.DataFrame | None = None):
        """-> (points frame with the query's static columns, {point_id: reference node id})."""
        pts = (static if static is not None else static_mod.point_static(self.grids, lat, lon)).copy()
        pid = point_id or QUERY_ID
        pts["point_id"] = pid
        return pts, {pid: fe.nearest_node_id(self.nodes, lat, lon)}

    # ------------------------------------------------------------------ tables
    def _rows(self, run: RunArrays, pts: pd.DataFrame):
        sampler = base.Sampler(pts["lat"].to_numpy(), pts["lon"].to_numpy())
        ids = pts["point_id"].to_numpy()
        derived = ext.derive_all(run.ext, self.terrain)
        p1 = dataset.merge_ext(base.station_step_rows(run.run_date, run.base, None, ids, sampler),
                               ext.station_step_ext(run.run_date, run.ext, self.terrain, ids, sampler, derived), KEYS1, how="inner")
        p2 = dataset.merge_ext(base.point_day_rows(run.run_date, run.base, ids, sampler),
                               ext.point_day_ext(run.run_date, run.ext, self.terrain, ids, sampler, derived), KEYS2, how="inner")
        return p1, p2

    def _anomalise(self, frame: pd.DataFrame, ref_map: dict) -> pd.DataFrame:
        return fe.apply_climatology(frame, self.clim, "day", ref_map)

    def step_table(self, run: RunArrays, lat, lon, point_id=None, static=None) -> pd.DataFrame:
        """Hourly-step features (thunderstorm, fog, wind, rain_3h, dust, hourly ranges): one row per forecast step."""
        pts, ref = self.locate(lat, lon, point_id, static)
        p1, p2 = self._rows(run, pts)
        p2a = self._anomalise(p2, ref)
        anoms = {f"{c}_anom": f"day_{c}_anom" for c in fe.STEP_DAY_ANOM_COLS if f"{c}_anom" in p2a}
        day_anom = p2a[KEYS2 + list(anoms)].rename(columns=anoms).rename(columns={"day_k": "day_k_valid"})
        p1s = p1.assign(day_k_valid=((p1["lead_h"] - 1) // 24).astype("int8"))
        p1s = p1s.merge(day_anom, on=["point_id", "run_date", "day_k_valid"], how="left").drop(columns=["day_k_valid"])
        table = fe.step_context(p1s)
        table["valid_time"] = pd.to_datetime(table["run_date"]) + pd.to_timedelta(table["lead_h"].astype(int), unit="h")
        table["valid_hour"] = table["valid_time"].dt.floor("h")
        table = fe.build_step_features(table, pts)
        table["month"] = table["valid_time"].dt.month
        table["lead_bucket"] = fe.lead_bucket(table["lead_h"])
        return table

    def station_day_table(self, run: RunArrays, lat, lon, point_id=None, static=None) -> pd.DataFrame:
        """IST-day features (Tmax/Tmin ranges, hot day, heat/cold wave): one row per forecast IST day 1..9."""
        pts, ref = self.locate(lat, lon, point_id, static)
        p1, _ = self._rows(run, pts)
        agg = self._anomalise(dataset.ist_day_aggregates(p1), ref)
        table = dataset.filter_complete_ist_days(agg)
        table = fe.build_day_features(table, pts, "ist_day")
        table["month"] = pd.to_datetime(table["ist_day"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"day{d}")
        return table

    def rain_day_table(self, run: RunArrays, lat, lon, point_id=None, static=None) -> pd.DataFrame:
        """UTC-day features for the rain exceedance curve: one row per forecast day 0..9."""
        pts, ref = self.locate(lat, lon, point_id, static)
        _, p2 = self._rows(run, pts)
        table = self._anomalise(p2, ref)
        table = table[table["rain_buckets"] >= 4].copy()
        table["valid_date"] = pd.to_datetime(table["run_date"]) + pd.to_timedelta(table["day_k"].astype(int), unit="D")
        table = fe.build_day_features(table, pts, "valid_date")
        table["month"] = pd.to_datetime(table["valid_date"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"day{d}")
        return table

    def window_table(self, run: RunArrays, lat, lon, window: int, point_id=None, static=None) -> pd.DataFrame:
        """Multi-day window features (union / total of rain over `window` days): one row per start day."""
        pts, _ = self.locate(lat, lon, point_id, static)
        _, p2 = self._rows(run, pts)
        table = dataset.window_feature_table(p2, window, dataset.window_day_columns(p2.columns))
        table = fe.build_day_features(table, pts, "valid_date")
        table["month"] = pd.to_datetime(table["valid_date"]).dt.month
        table["lead_bucket"] = table["day_k"].map(lambda d: f"start_day{d}")
        return table
