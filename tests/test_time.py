
from app.services.time_parser import parse_time_window
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
def test_tomorrow_afternoon():
    now = datetime(2026,9,1,10,0, tzinfo=IST)
    vf, vt, hor, conf = parse_time_window("tomorrow afternoon", now)
    assert vt.hour == 18
    assert hor == "short"
def test_next_week():
    now = datetime(2026,9,1,10,0, tzinfo=IST)
    vf, vt, hor, conf = parse_time_window("next week", now)
    assert hor in ("medium","climate")
def test_explicit_date():
    now = datetime(2026,9,1,10,0, tzinfo=IST)
    vf, vt, hor, conf = parse_time_window("2024-01-15", now)
    assert vf.year == 2024
