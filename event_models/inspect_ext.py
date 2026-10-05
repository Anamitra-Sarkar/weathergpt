"""Print per-column NaN rate and value range of the extension tables (what a smoke run actually produced)."""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

INPUT = "/kaggle/input"


def describe(frame: pd.DataFrame, skip: set) -> dict:
    rows = {}
    for col in frame.columns:
        if col in skip:
            continue
        x = frame[col].astype("float64")
        rows[col] = {"nan": round(float(x.isna().mean()), 3),
                     "min": None if x.notna().sum() == 0 else round(float(np.nanmin(x)), 3),
                     "p50": None if x.notna().sum() == 0 else round(float(np.nanmedian(x)), 3),
                     "max": None if x.notna().sum() == 0 else round(float(np.nanmax(x)), 3)}
    return rows


def main():
    report = {}
    for name, pattern in (("p1x", "p1x_station_steps_*.parquet"), ("p2x", "p2x_point_days_*.parquet"),
                          ("points_ext", "points_ext.parquet")):
        files = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        skip = {"point_id", "run_date"}
        info = {"rows": len(frame), "n_cols": len(frame.columns) - len(skip & set(frame.columns)),
                "columns": describe(frame, skip)}
        info["all_nan_columns"] = [c for c, v in info["columns"].items() if v["nan"] >= 0.999]
        info["mostly_nan_columns"] = {c: v["nan"] for c, v in info["columns"].items() if 0.2 < v["nan"] < 0.999}
        report[name] = info
    print("INSPECT_BEGIN")
    print(json.dumps(report, indent=0, default=str))
    print("INSPECT_END")


if __name__ == "__main__":
    main()
