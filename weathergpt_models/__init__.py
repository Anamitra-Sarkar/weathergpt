"""WeatherGPT trained models — standalone inference package.

Depends only on numpy, torch, transformers, lightgbm and scikit-learn.  It does
not import `app`, FastAPI, or any part of the serving framework, so the
framework can consume it without inheriting the training stack's shape.

    from weathergpt_models import ModelRegistry

    registry = ModelRegistry.from_hub("<repo-id>")      # or .from_dir(path)
    print(registry.status())                            # what loaded, and why not

    registry.field_mapper.map_field("APCP", unit="kg m-2",
                                    time_range_text="0-3 hour acc fcst")
    registry.intent.parse("kal Bhandara me baarish hogi kya?")
    registry.mos.correct("temperature_2m", forecasts={...}, context={...})
    registry.calibration.exceedance_probability("precipitation", 5.0, ...)
    registry.trust_ranker.rank("temperature_2m", candidates={...}, context={...})

Every artifact passes a metrics contract before it loads.  An artifact whose
`metrics.json` lacks provenance, or whose headline metric does not beat the
baseline it was measured against, is refused and the corresponding attribute
stays `None` — the caller falls back to its deterministic path and
`registry.status()` says exactly which gate failed.  This exists because this
project previously shipped a checkpoint fitted to `np.random.randn` alongside a
`metrics.json` claiming it was trained on real matched pairs.
"""
from weathergpt_models.registry import ModelRegistry, GateResult  # noqa: F401
from weathergpt_models.types import (  # noqa: F401
    FieldMapping,
    ParsedQuery,
    CorrectedForecast,
    CalibratedProbability,
    RankedSource,
    Slot,
)

__all__ = ["ModelRegistry", "GateResult", "FieldMapping", "ParsedQuery",
           "CorrectedForecast", "CalibratedProbability", "RankedSource", "Slot"]
__version__ = "1.0.0"
