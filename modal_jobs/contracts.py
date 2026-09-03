"""Dataset contracts.  A build that violates one of these writes nothing.

Every check here exists because its absence already cost this project a dataset.
The headline one: the previous `matched_pairs.csv` shipped 4800 rows in which
the "forecast" and the "observation" columns were byte-identical, so the
regression target was zero everywhere and the model reported a beautiful RMSE
for having learned nothing.  Four lines would have caught it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ContractViolation(AssertionError):
    pass


@dataclass
class ContractReport:
    name: str
    checks: list = field(default_factory=list)

    def check(self, label: str, ok: bool, detail: str = "", fatal: bool = True) -> None:
        self.checks.append({"check": label, "ok": bool(ok), "detail": detail, "fatal": fatal})
        if not ok and fatal:
            raise ContractViolation(f"[{self.name}] {label}: {detail}")

    @property
    def passed(self) -> bool:
        return all(item["ok"] for item in self.checks if item["fatal"])

    def summary(self) -> dict:
        failed = [item for item in self.checks if not item["ok"]]
        return {"dataset": self.name, "passed": self.passed,
                "n_checks": len(self.checks), "failed": failed, "checks": self.checks}


def check_forecast_truth_corpus(frame, *, name: str, pairs: list, min_rows: int = 10_000):
    """`pairs` is a list of (forecast_column, truth_column)."""
    import numpy as np

    report = ContractReport(name)
    report.check("non_empty", len(frame) >= min_rows, f"{len(frame)} rows < {min_rows}")

    seen_truth = set()
    for forecast_column, truth_column in pairs:
        if forecast_column not in frame.columns or truth_column not in frame.columns:
            report.check(f"columns_present::{forecast_column}", False,
                         f"missing {forecast_column} or {truth_column}")
            continue
        truth = frame[truth_column].to_numpy(dtype="float64")
        if truth_column not in seen_truth:
            seen_truth.add(truth_column)
            finite = float(np.isfinite(truth).mean())
            report.check(f"truth_present::{truth_column}", finite > 0.5,
                         f"only {finite:.1%} of {truth_column} is finite — a truth source "
                         f"returning nulls silently produces an untrainable target")
        forecast = frame[forecast_column].to_numpy(dtype="float64")
        both = np.isfinite(forecast) & np.isfinite(truth)
        if both.sum() < 100:
            report.check(f"pair_overlap::{forecast_column}", False,
                         f"only {int(both.sum())} rows where forecast and truth are both finite")
            continue
        difference = forecast[both] - truth[both]
        max_abs = float(np.abs(difference).max())
        report.check(f"non_degenerate::{forecast_column}", max_abs > 1e-3,
                     f"max |forecast - truth| = {max_abs:.2e}; the two columns are the same "
                     f"data, so the learning target is identically zero")
    return report


def check_lead_time_signal(frame, *, report: ContractReport, lead_column: str,
                           forecast_column: str, truth_column: str) -> None:
    """Forecast error must grow with lead time.  If it does not, the lead column
    is not a forecast age — which is exactly what `lead_hours = i % 72` was."""
    import numpy as np

    subset = frame[[lead_column, forecast_column, truth_column]].dropna()
    if len(subset) < 1000:
        report.check("lead_signal", False, f"only {len(subset)} paired rows", fatal=False)
        return
    error = (subset[forecast_column] - subset[truth_column]).abs()
    grouped = error.groupby(subset[lead_column]).mean().sort_index()
    if len(grouped) < 3:
        report.check("lead_signal", False, f"only {len(grouped)} distinct leads", fatal=False)
        return
    leads = grouped.index.to_numpy(dtype="float64")
    values = grouped.to_numpy(dtype="float64")
    slope = float(np.polyfit(leads, values, 1)[0])
    report.check("lead_error_grows_with_lead", slope > 0,
                 f"mean |error| slope vs lead = {slope:.6f} per unit lead; a non-positive "
                 f"slope means the lead column carries no forecast-age information. "
                 f"profile={dict(zip(leads.tolist(), np.round(values, 3).tolist()))}",
                 fatal=False)


def check_label_corpus(frame, *, name: str, label_column: str, split_column: str,
                       key_column: str, min_rows: int = 1000,
                       max_duplicate_fraction: float = 0.05):
    report = ContractReport(name)
    report.check("non_empty", len(frame) >= min_rows, f"{len(frame)} rows < {min_rows}")

    duplicated = float(frame.duplicated(subset=[key_column, label_column]).mean())
    report.check("low_duplication", duplicated <= max_duplicate_fraction,
                 f"{duplicated:.1%} duplicate (key, label) rows — duplicates straddling a "
                 f"split turn reported accuracy into a memorisation score")

    splits = set(frame[split_column].unique())
    report.check("has_splits", {"train", "val"} <= splits, f"splits present: {sorted(splits)}")

    keys = {split: set(part[key_column]) for split, part in frame.groupby(split_column)}
    for left in keys:
        for right in keys:
            if left >= right:
                continue
            overlap = keys[left] & keys[right]
            report.check(f"split_disjoint::{left}|{right}", not overlap,
                         f"{len(overlap)} keys in both splits, e.g. {sorted(overlap)[:5]}")

    counts = frame[label_column].value_counts()
    report.check("more_than_one_class", len(counts) > 1, f"only {len(counts)} class present")
    report.check("majority_not_total", counts.iloc[0] / len(frame) < 0.98,
                 f"majority class {counts.index[0]} is {counts.iloc[0] / len(frame):.1%} of rows")
    return report


def check_no_unit_contradiction(frame, *, report: ContractReport,
                                variable_column: str = "canonical_variable",
                                unit_family_column: str = "unit_family") -> None:
    """A temperature can never be measured in millimetres.

    The previous corpus contained 64 rows labelled `temperature_*` whose field
    name carried a `(mm)` suffix, because the unit suffix was sampled without
    consulting the label.
    """
    temperature = {"temperature_2m", "temperature_max", "temperature_min", "dewpoint_2m",
                   "apparent_temperature", "soil_temperature", "sea_surface_temperature"}
    bad = frame[frame[variable_column].isin(temperature)
                & frame[unit_family_column].isin(["length", "precip_depth", "precip_rate"])]
    report.check("no_temperature_in_depth_units", len(bad) == 0,
                 f"{len(bad)} temperature rows carry a depth unit, e.g. "
                 f"{bad.head(3)[['raw_field', 'unit']].to_dict('records') if len(bad) else []}")

    precipitation = {"precipitation_amount", "precipitation_rate", "snowfall_amount"}
    bad = frame[frame[variable_column].isin(precipitation)
                & (frame[unit_family_column] == "temperature")]
    report.check("no_precipitation_in_temperature_units", len(bad) == 0,
                 f"{len(bad)} precipitation rows carry a temperature unit")
