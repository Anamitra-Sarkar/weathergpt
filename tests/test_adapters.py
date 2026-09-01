
import pytest
from app.adapters.registry import REGISTRY

def test_registry_has_all():
    assert "OPEN_METEO" in REGISTRY
    assert "IMD" in REGISTRY
    assert "GFS" in REGISTRY

def test_grib2_unavailable():
    import asyncio
    adapter = REGISTRY["GFS"]
    result = asyncio.run(adapter.health_check())
    assert result["available"] == False
    assert "eccodes" in result["reason"] or "isolated" in result["reason"]

def test_imd_without_key():
    import asyncio, os
    os.environ.pop("IMD_API_KEY", None)
    adapter = REGISTRY["IMD"]
    result = asyncio.run(adapter.health_check())
    assert result["available"] == False
    assert "IMD_API_KEY" in result["reason"]
