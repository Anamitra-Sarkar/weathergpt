"""Adapter registry — centralizes health and source selection."""
from __future__ import annotations
from typing import Dict, List
from app.adapters.open_meteo_forecast import OpenMeteoForecastAdapter
from app.adapters.open_meteo_historical import OpenMeteoHistoricalAdapter
from app.adapters.open_meteo_ensemble import OpenMeteoEnsembleAdapter
from app.adapters.cap_adapter import CapAdapter
from app.adapters.nasa_power import NasaPowerAdapter
from app.adapters.imd_adapter import ImdAdapter
from app.adapters.grib2_adapter import Grib2Adapter

REGISTRY = {
    "OPEN_METEO": OpenMeteoForecastAdapter(),
    "ERA5": OpenMeteoHistoricalAdapter(),
    "GEFS": OpenMeteoEnsembleAdapter(),
    "CAP": CapAdapter(),
    "NASA_POWER": NasaPowerAdapter(),
    "IMD": ImdAdapter(),
    "GFS": Grib2Adapter(),
}

async def health_all() -> Dict[str, Dict]:
    out = {}
    for name, adapter in REGISTRY.items():
        try:
            out[name] = await adapter.health_check()
        except Exception as e:
            out[name] = {"available": False, "reason": str(e)}
    return out

def get_adapter(name: str):
    return REGISTRY.get(name)
