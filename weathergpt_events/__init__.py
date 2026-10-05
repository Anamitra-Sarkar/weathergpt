"""Event / range / rain-curve models: inference interface (see README.md).

Imports only numpy / pandas / scipy / lightgbm; the feature code is the SAME code the models were trained with
(`event_models`), never a re-implementation.
"""
import os

# The collectors read the lead set at import time.  Serving always wants the full 10-day horizon (52 steps).
os.environ.setdefault("GFS_LEADSET", "all")

from weathergpt_events.registry import EventRegistry  # noqa: E402,F401
