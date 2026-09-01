
from app.services.fusion import fuse
from app.schemas.ceo import CanonicalEvidenceObject, Geometry, Provenance
from datetime import datetime, timezone
def test_fusion_best_by_var():
    now=datetime.now(timezone.utc)
    ceos=[
        CanonicalEvidenceObject(source="OPEN_METEO", evidence_class="forecast", variable="temperature_2m", value=30, unit="C", statistic="instant", geometry=Geometry(type="Point", coordinates=[79,21]), valid_from=now, valid_to=now, provenance=Provenance(original_source="OPEN_METEO", transformations=[])),
        CanonicalEvidenceObject(source="IMD", evidence_class="forecast", variable="temperature_2m", value=31, unit="C", statistic="instant", geometry=Geometry(type="Point", coordinates=[79,21]), valid_from=now, valid_to=now, provenance=Provenance(original_source="IMD", transformations=[])),
    ]
    result=fuse(ceos, 21,79, now)
    assert "temperature_2m" in result["best_by_var"]
    assert result["fusion_metadata"]["scored_count"]==2
