import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from app.main import app
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance


def _evidence():
    now = datetime.now(timezone.utc) + timedelta(days=1)
    geometry = Geometry(type="Point", coordinates=[79.0882, 21.1458])
    provenance = Provenance(original_source="fixture", transformations=["fixture test"])
    return [
        CanonicalEvidenceObject(source="OPEN_METEO", evidence_class="forecast", variable="precipitation_amount", value=2.0, unit="mm", statistic="accumulation", geometry=geometry, valid_from=now, valid_to=now, accumulation_window_hours=1, provenance=provenance),
        CanonicalEvidenceObject(source="OPEN_METEO", evidence_class="forecast", variable="precipitation_probability", value=.6, probability=.6, unit="probability", statistic="probability", geometry=geometry, valid_from=now, valid_to=now, provenance=provenance),
    ]


async def _post(path, body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=body)


def test_unknown_location_is_structured_error():
    response = asyncio.run(_post("/wio/query", {"question": "weather tomorrow", "location": {"raw": "AtlantisXYZ"}}))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "LOCATION_NOT_FOUND"


def test_wio_is_evidence_backed(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return _evidence(), {"sources": {"fixture": {"status": "ok"}}, "partial": False}
    monkeypatch.setattr("app.main.retrieve", fake_retrieve)
    response = asyncio.run(_post("/wio/query", {"question": "Will it rain in Nagpur tomorrow?"}))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["wio"]["evidence"]
    assert data["agents"][-2]["agent_name"] == "reviewer"
    assert data["agents"][-2]["status"] == "success"
