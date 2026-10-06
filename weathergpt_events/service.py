"""A small, framework-free service around the engine: newest-run cache, validated tool calls, readable summaries.

    service = ForecastService.live(load_engine("Arko007/weathergpt-events"), cache_dir="run_cache")
    service.forecast(28.61, 77.21, targets=["rain_curve", "tmax_range"], horizon_days=3)
    service.call_tool("thunderstorm", {"lat": 19.08, "lon": 72.88})        # what an orchestrator LLM would emit

Every failure is a structured `{"available": False, "reason": ...}`; nothing here invents a number.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Callable

from weathergpt_events import live
from weathergpt_events import tools
from weathergpt_events.engine import EventEngine
from weathergpt_events.features import RunArrays

MAX_HORIZON_DAYS = 10
CONTROL_TOOLS = ("catalogue", "status")


class ToolCallError(ValueError):
    """The orchestrator asked for something the catalogue does not offer (unknown tool, bad or extra arguments)."""


class ForecastService:
    def __init__(self, engine: EventEngine, run_provider: Callable[[], RunArrays | None], ttl_s: float = 900.0,
                 clock: Callable[[], float] = time.monotonic):
        """`run_provider()` returns the newest fresh run (or None when NOAA has none fresh enough).  It is called at most
        once per `ttl_s`; the result (including "none") is cached so a burst of questions costs one availability check."""
        self.engine, self._provider, self._ttl, self._clock = engine, run_provider, ttl_s, clock
        self._cached: tuple | None = None          # (checked_at, run or None)

    @classmethod
    def live(cls, engine: EventEngine, cache_dir: str, workers: int = 8, ttl_s: float = 900.0) -> "ForecastService":
        store = live.RunStore(cache_dir, workers=workers)

        def newest() -> RunArrays | None:
            run_date = store.latest(datetime.now(timezone.utc))
            return None if run_date is None else store.load(run_date)
        return cls(engine, newest, ttl_s)

    # ------------------------------------------------------------------ run cache
    def current_run(self) -> RunArrays | None:
        now = self._clock()
        if self._cached is None or now - self._cached[0] > self._ttl:
            self._cached = (now, self._provider())
        return self._cached[1]

    # ------------------------------------------------------------------ queries
    def forecast(self, lat: float, lon: float, targets: list | None = None, horizon_days: int = MAX_HORIZON_DAYS,
                 include_unvalidated: bool = False) -> dict:
        run = self.current_run()
        if run is None:
            return {"available": False, "reason": f"no fresh GFS/GEFS run is published (older than {live.MAX_RUN_AGE_H} h or not complete); "
                                                  "refusing to present a stale forecast as current"}
        return self.engine.forecast(float(lat), float(lon), run, targets=targets, horizon_days=int(horizon_days),
                                    include_unvalidated=include_unvalidated)

    def catalogue(self) -> list:
        return tools.catalogue(self.engine.registry)

    def status(self) -> dict:
        run = self.current_run()
        return {"run": None if run is None else run.run_date.isoformat(), "models": self.engine.registry.status(),
                "served": sorted(self.engine.registry.models)}

    # ------------------------------------------------------------------ orchestrator interface
    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Execute one tool call exactly as `catalogue()` advertises it.  Raises ToolCallError for anything off-catalogue."""
        arguments = dict(arguments or {})
        if name == "catalogue" and not arguments:
            return {"tools": self.catalogue()}
        if name == "status" and not arguments:
            return self.status()
        entry = next((c for c in self.catalogue() if c["name"] == name), None)
        if entry is None:
            raise ToolCallError(f"unknown tool {name!r}; available: {[c['name'] for c in self.catalogue() if c['available']] + list(CONTROL_TOOLS)}")
        if not entry["available"]:
            return {"available": False, "reason": f"tool {name!r} is not served: {entry['why_not']}"}
        extra = set(arguments) - {"lat", "lon", "horizon_days"}
        if extra:
            raise ToolCallError(f"unexpected arguments {sorted(extra)} for {name!r}")
        lat, lon = _number(arguments, "lat"), _number(arguments, "lon")
        horizon = arguments.get("horizon_days", MAX_HORIZON_DAYS)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON_DAYS:
            raise ToolCallError(f"horizon_days must be an integer from 1 to {MAX_HORIZON_DAYS}, got {horizon!r}")
        return self.forecast(lat, lon, targets=[name], horizon_days=horizon)


def _number(arguments: dict, key: str) -> float:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolCallError(f"{key} must be a finite number, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------- human-readable summary
def _fmt(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def summarise(result: dict, max_rows: int = 3) -> str:
    """Compact text for a forecast dict (what the CLI prints and the notebook shows)."""
    if not result.get("available", False):
        return f"NOT AVAILABLE: {result.get('reason')}"
    loc = result["location"]
    lines = [f"run {result['run']}  at ({loc['lat']}, {loc['lon']})  land fraction {loc.get('land_fraction')}"
             + ("  [coastal cell]" if loc.get("coastal") else "")]
    for name, res in result["targets"].items():
        if not res.get("available"):
            lines.append(f"  {name:22s} not available: {res.get('reason')}")
            continue
        ok = [e for e in res["entries"] if e["validated"]]
        if not ok:
            reasons = {e.get("note") for e in res["entries"]}
            lines.append(f"  {name:22s} no validated answer in zone '{res['zone']}' ({'; '.join(sorted(r for r in reasons if r))[:110]})")
            continue
        lines.append(f"  {name:22s} zone {res['zone']}, {len(ok)}/{len(res['entries'])} entries validated")
        for e in ok[:max_rows]:
            when = e.get("valid_time_utc") or e.get("ist_day") or e.get("utc_day") or e.get("window_start_utc_day")
            if "probability" in e:
                body = f"P = {_fmt(e['probability'], 3)}"
            elif "q50" in e:
                body = f"{_fmt(e['q10'], 1)} / {_fmt(e['q50'], 1)} / {_fmt(e['q90'], 1)}  (80% range {_fmt(e['lo'], 1)} .. {_fmt(e['hi'], 1)})"
            elif "p_any_rain" in e:
                amt = e["amount_if_wet_mm"]
                body = (f"P(>=1 mm) = {_fmt(e['p_any_rain'], 3)}, P(>=2.5 mm) = {_fmt(e['p_rainy_day'], 3)}, "
                        f"amount if wet (10/50/90) = {_fmt(amt['p10'], 1)} / {_fmt(amt['p50'], 1)} / {_fmt(amt['p90'], 1)} mm")
            else:
                amt = e.get("amount_if_wet_mm") or {}
                body = f"amount if wet (10/50/90) = {_fmt(amt.get('p10'), 1)} / {_fmt(amt.get('p50'), 1)} / {_fmt(amt.get('p90'), 1)} mm"
            lines.append(f"      {when}  {body}")
        if res.get("thresholds_without_skill_mm"):
            lines.append(f"      (no held-out skill at thresholds {res['thresholds_without_skill_mm']} mm: treat those tail values as unvalidated)")
    return "\n".join(lines)
