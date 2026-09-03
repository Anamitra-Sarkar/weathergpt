"""Backwards-compatible shim.

The field taxonomy is model-layer knowledge — it defines the label space the
M1 mapper predicts into — so it lives in the standalone `weathergpt_models`
package and is re-exported here for existing `app.services` imports.  One
definition, two import paths; never two copies.
"""
from weathergpt_models.taxonomy import (  # noqa: F401
    CANONICAL_VARIABLES,
    EVIDENCE_CLASSES,
    GRIB_STAT_PROCESSING,
    LABEL_DESCRIPTIONS,
    STATISTICS,
    VARIABLE_SPECS,
    VERTICAL_LEVELS,
    FieldLabel,
    VariableSpec,
    classify_native_field,
    parse_accumulation_hours,
    parse_vertical_level,
    unit_family,
)

__all__ = [
    "CANONICAL_VARIABLES", "EVIDENCE_CLASSES", "GRIB_STAT_PROCESSING",
    "LABEL_DESCRIPTIONS", "STATISTICS", "VARIABLE_SPECS", "VERTICAL_LEVELS",
    "FieldLabel", "VariableSpec", "classify_native_field",
    "parse_accumulation_hours", "parse_vertical_level", "unit_family",
]
