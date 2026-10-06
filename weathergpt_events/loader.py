"""One call from "a folder (or a Hugging Face repo id) of published artifacts" to a working engine.

Expected layout (what `event_models/publish_hf.py` publishes):

    models/<target>/{model*.txt, calibrator.json | interval.json, features.json, metrics.json}
    static_grids.npz      GFS orography / land mask derived grids (see static.py)
    points.parquet        the 2,709 training points (nodes + stations); gives the climatology reference nodes and the coastal rule
    climatology_day.parquet

Nothing here talks to the weather servers: that is `live.RunStore`'s job.  A Hugging Face repo id is downloaded with
`huggingface_hub` (imported lazily); pass `token` for a private repo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from weathergpt_events import static as static_mod
from weathergpt_events.engine import EventEngine
from weathergpt_events.features import FeatureBuilder
from weathergpt_events.registry import EventRegistry

REQUIRED = ("static_grids.npz", "points.parquet", "climatology_day.parquet")


def resolve_source(source: str | Path, token: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """A local folder is used as is; anything else is treated as a Hugging Face model repo id and downloaded."""
    path = Path(str(source))
    if path.is_dir():
        return path
    if "/" not in str(source):
        raise FileNotFoundError(f"{source!r} is neither a folder nor a Hugging Face repo id (owner/name)")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(source), token=token, local_dir=str(cache_dir) if cache_dir else None,
                                  allow_patterns=["models/*", "*.npz", "*.parquet", "README.md"]))


def check_layout(root: Path) -> None:
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if not (root / "models").is_dir():
        missing.append("models/")
    if missing:
        raise FileNotFoundError(f"{root} is not a WeatherGPT events artifact folder; missing: {missing}")


def load_engine(source: str | Path, token: str | None = None, cache_dir: str | Path | None = None) -> EventEngine:
    root = resolve_source(source, token, cache_dir)
    check_layout(root)
    registry = EventRegistry.from_dir(root / "models")
    if not registry.models:
        raise RuntimeError("no model passed the admission gate: " + "; ".join(f"{r['target']}: {r['reason']}" for r in registry.status()))
    builder = FeatureBuilder(static_mod.load_static_grids(root / "static_grids.npz"),
                             pd.read_parquet(root / "climatology_day.parquet"), pd.read_parquet(root / "points.parquet"))
    return EventEngine(registry, builder)
