"""Fetch (and cache) the latest GFS + GEFS run on the India window, with the collectors' own fetch code.

The cache holds one run per day on disk (float32 grids for 52 steps, ~300 MB), so every query after the first one that
day samples local arrays in milliseconds.  Network access happens only here.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from event_models import collect_gfs as base
from event_models import collect_gfs_ext as ext
from weathergpt_events import static as static_mod
from weathergpt_events.features import RunArrays

MAX_RUN_AGE_H = 54            # a 00Z run is published ~5 h later and the next one ~24 h after that; if NOAA is late the
                              # newest run can be ~2 days old.  Beyond this the forecast is stale: refuse, don't serve it as current.


def run_age_hours(run_date: date, now: datetime) -> float:
    """Hours since the run's 00Z initialisation time."""
    return (now - datetime.combine(run_date, datetime.min.time(), tzinfo=timezone.utc)).total_seconds() / 3600


def run_is_fresh(run_date: date, now: datetime) -> bool:
    return 0 <= run_age_hours(run_date, now) <= MAX_RUN_AGE_H


def _flatten(prefix: str, arrays: dict) -> dict:
    return {f"{prefix}|{lead}|{name}": np.asarray(arr) for lead, fields in arrays.items() for name, arr in fields.items()}


def _unflatten(blob) -> tuple:
    base_arrays, ext_arrays = {}, {}
    for key in blob.files:
        prefix, lead, name = key.split("|", 2)
        target = base_arrays if prefix == "b" else ext_arrays
        value = blob[key]
        target.setdefault(int(lead), {})[name] = value.item() if value.ndim == 0 else value
    return base_arrays, ext_arrays


class RunStore:
    def __init__(self, cache_dir: str | Path, workers: int = 8):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def _pool(self):
        base.ensure_eccodes()
        return cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("fork"), initializer=base._init_worker)

    def available(self, run_date: date, pool) -> bool:
        """A run is usable once its LAST file (f240) is published for both GFS and GEFS."""
        gfs = dict(pool.map(base.idx_task, [(run_date, max(base.LEADS))]))
        gefs = dict(pool.map(ext.gefs_idx_task, [("geavg", run_date, max(base.LEADS)), ("gespr", run_date, max(base.LEADS))]))
        return all(v is not None for v in gfs.values()) and all(v is not None for v in gefs.values())

    def latest(self, now: datetime | None = None) -> date | None:
        now = now or datetime.now(timezone.utc)
        with self._pool() as pool:
            for back in range(0, 4):
                candidate = (now - timedelta(days=back)).date()
                if self.available(candidate, pool):
                    return candidate if run_is_fresh(candidate, now) else None     # newest published run, if it is not stale
        return None

    def load(self, run_date: date, keep: int = 3) -> RunArrays:
        path = self.cache / f"run_{run_date:%Y%m%d}.npz"
        if path.exists():
            b, e = _unflatten(np.load(path))
            return RunArrays(run_date, b, e)
        with self._pool() as pool:
            base_arrays, base_stats = base.fetch_run(pool, run_date)
            ext_arrays, ext_stats = ext.fetch_ext_run(pool, run_date)
        if base_arrays is None or ext_arrays is None:
            raise RuntimeError(f"run {run_date} is not available (GFS: {base_stats}, GEFS/ext: {ext_stats})")
        np.savez(path, **_flatten("b", base_arrays), **_flatten("e", ext_arrays))
        for old in sorted(self.cache.glob("run_*.npz"))[:-keep]:      # keep the last few runs only
            old.unlink()
        return RunArrays(run_date, base_arrays, ext_arrays)


def fetch_static_arrays(run_date: date) -> dict:
    """One-off: the GFS orography and land-sea mask on the window (used to build the static grids)."""
    base.ensure_eccodes()
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=ctx, initializer=base._init_worker) as pool:
        # idx_task needs the worker's HTTP client: it must run in the pool, not in this process
        (lead, found), = pool.map(base.idx_task, [(run_date, base.LEADS[0])])
        if found is None:
            raise RuntimeError(f"GFS index for {run_date} f{base.LEADS[0]:03d} is not published")
        url, rows = found
        picked = base.pick(rows, base.STATIC)
        tasks = [(lead, name, url, s, e, d) for name, (s, e, d) in picked.items()]
        out = {name: window for _, name, window, _ in pool.map(base.msg_task, tasks)}
    if "hgt" not in out or "land" not in out:
        raise RuntimeError(f"orography / land mask missing from {url}")
    return out


def build_and_save_static(run_date: date, path: str | Path) -> dict:
    arrays = fetch_static_arrays(run_date)
    grids = static_mod.build_static_grids(arrays["land"], arrays["hgt"])
    static_mod.save_static_grids(grids, path)
    return grids
