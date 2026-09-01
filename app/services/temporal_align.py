from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from app.schemas.ceo import CanonicalEvidenceObject

IST = timezone(timedelta(hours=5, minutes=30))

def to_utc(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def overlaps(ev: CanonicalEvidenceObject, q_from: datetime, q_to: datetime) -> bool:
    if ev.valid_from is None or ev.valid_to is None:
        # fallback to issued/observed
        t = ev.issued_at or ev.observed_at
        if t is None:
            return True  # keep if no time at all
        t = to_utc(t)
        return q_from <= t <= q_to
    return not (to_utc(ev.valid_to) < q_from or to_utc(ev.valid_from) > q_to)

def filter_by_window(evs: List[CanonicalEvidenceObject], q_from: datetime, q_to: datetime) -> List[CanonicalEvidenceObject]:
    return [e for e in evs if overlaps(e, q_from, q_to)]

def staleness_hours(ev: CanonicalEvidenceObject, now: datetime) -> float:
    t = ev.issued_at or ev.model_initialization_time or ev.observed_at or ev.ingested_at
    if t is None:
        return 999
    return (to_utc(now) - to_utc(t)).total_seconds() / 3600

def is_stale(ev: CanonicalEvidenceObject, now: datetime, max_age_hours: float = 48) -> bool:
    return staleness_hours(ev, now) > max_age_hours
