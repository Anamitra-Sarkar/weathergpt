from app.decoders.imd_json import decode_city_forecast, decode_warning, decode_nowcast
from app.decoders.cap_decoder import decode_cap_xml

def test_imd_city():
    rec = {"city":"Nagpur","lat":21.14,"lon":79.08,"forecast_date":"2026-09-01T00:00:00+05:30","rainfall":12,"temp_max":32,"issued_at":"2026-08-31T06:00:00+05:30"}
    ceos = decode_city_forecast(rec)
    assert any(c.variable=="precipitation_amount" for c in ceos)
    assert any(c.accumulation_window_hours==24 for c in ceos)

def test_imd_warning():
    rec = {"district":"Nagpur","hazard":"heavy rainfall","colour":"orange","valid_from":"2026-09-01T00:00:00+05:30","valid_to":"2026-09-01T12:00:00+05:30","issue_time":"2026-08-31T06:00:00+05:30"}
    ceos = decode_warning(rec)
    assert ceos[0].warning_severity=="orange"
    assert ceos[0].evidence_class=="warning"

def test_cap():
    xml = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<alert xmlns=\"urn:oasis:names:tc:emergency:cap:1.2\">\n<identifier>TEST-1</identifier><sender>imd@test</sender><sent>2026-08-31T06:00:00+05:30</sent><status>Actual</status><msgType>Alert</msgType>\n<info><event>Heavy Rainfall</event><severity>Severe</severity><effective>2026-09-01T00:00:00+05:30</effective><expires>2026-09-01T12:00:00+05:30</expires><area><areaDesc>Nagpur</areaDesc></area></info></alert>"""
    ceos = decode_cap_xml(xml)
    assert len(ceos)==1
    assert ceos[0].warning_severity=="orange"
