"""M5 inference — which source should be believed, here and now."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weathergpt_models.features import MODELS
from weathergpt_models.types import RankedSource

HEURISTIC_AUTHORITY_ORDER = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless")


class TrustRanker:
    """Orders candidate NWP sources for a specific location, time and lead.

    The hand-tuned alternative — a static authority table — necessarily produces
    one global ordering, so the same model wins in Leh in January and in Kochi
    in July.  This ranker conditions on lead time, season, elevation, terrain,
    inter-model spread and each source's measured historical skill at that
    location.  `metrics.json` records how often following it picks the source
    that was actually closest to truth, against how often the fixed order does.
    """

    def __init__(self, directory: str | Path):
        import lightgbm as lgb

        self.directory = Path(directory)
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.algorithm_version = metrics.get("algorithm_version", "m5")
        self.metrics = metrics
        self.sources = tuple(metrics.get("candidate_sources", MODELS))

        self._boosters = {}
        for path in sorted(self.directory.glob("lambdamart_*.txt")):
            variable = path.stem[len("lambdamart_"):]
            self._boosters[variable] = lgb.Booster(model_file=str(path))

        self._skill = metrics.get("global_source_skill", {})

    def supports(self, variable: str) -> bool:
        return variable in self._boosters

    def rank(self, variable: str, *, candidates: dict, context: dict,
             historical_mae: dict | None = None) -> list:
        """`candidates` maps source name to its forecast value.

        `historical_mae` optionally supplies each source's measured mean absolute
        error at this location; when absent the model falls back to the global
        per-source skill recorded at training time, which is the honest default
        for a location with no history yet.
        """
        if variable not in self._boosters:
            raise KeyError(f"{variable} was not trained; available: {sorted(self._boosters)}")

        historical_mae = historical_mae or {}
        values = np.array([candidates.get(source, np.nan) for source in self.sources],
                          dtype="float64")
        with np.errstate(invalid="ignore"):
            multi_model_mean = np.nanmean(values)
            spread = np.nanstd(values)
            span = np.nanmax(values) - np.nanmin(values)

        doy = float(context.get("doy", 180))
        month = int(context.get("month", 0)) or int((doy / 30.5) + 1)
        rows = []
        for position, source in enumerate(self.sources):
            onehot = [1.0 if i == position else 0.0 for i in range(len(self.sources))]
            fallback = float(self._skill.get(source, 1.0))
            location_skill = float(historical_mae.get(source, fallback))
            rows.append(onehot + [
                float(values[position]),
                float(values[position] - multi_model_mean),
                float(spread), float(span),
                location_skill, location_skill, fallback,
                float(len(self.sources) - HEURISTIC_AUTHORITY_ORDER.index(source)),
                float(context.get("lead_hours", 0.0)),
                float(context.get("lead_age_days", 0.0)),
                float(np.sin(2 * np.pi * doy / 365.25)),
                float(np.cos(2 * np.pi * doy / 365.25)),
                float(context.get("elevation_m", 0.0)) / 1000.0,
                float(context.get("lat", 0.0)), float(context.get("lon", 0.0)),
                1.0 if month in (6, 7, 8, 9) else 0.0,
            ])

        scores = self._boosters[variable].predict(np.nan_to_num(np.array(rows), nan=0.0))
        order = np.argsort(-scores)
        return [RankedSource(source=self.sources[int(index)], score=float(scores[int(index)]),
                             rank=rank, value=(float(values[int(index)])
                                               if np.isfinite(values[int(index)]) else None))
                for rank, index in enumerate(order)]

    def best_source(self, variable: str, **kwargs) -> RankedSource:
        return self.rank(variable, **kwargs)[0]
