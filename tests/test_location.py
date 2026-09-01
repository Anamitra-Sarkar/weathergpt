
import pytest
from app.services.location_resolver import resolve_location, LocationNotFoundError, LocationAmbiguousError

def test_pincode_ok():
    loc = resolve_location("440001")
    assert loc.district == "Nagpur"
    assert loc.pincode == "440001"

def test_unknown_pincode_raises():
    with pytest.raises(LocationNotFoundError):
        resolve_location("999999")

def test_gps():
    loc = resolve_location("21.14,79.08")
    assert loc.lat == 21.14

def test_ambiguous():
    # "pune" vs "pune" is single, but "a" matches many? Use "pune mumbai" to trigger ambiguous
    with pytest.raises(LocationAmbiguousError):
        resolve_location("pune mumbai")

def test_unknown_city_raises():
    with pytest.raises(LocationNotFoundError):
        resolve_location("AtlantisXYZ")
