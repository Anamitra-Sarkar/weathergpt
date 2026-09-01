from __future__ import annotations
from datetime import datetime, timedelta, timezone
import re

IST = timezone(timedelta(hours=5, minutes=30))

def parse_time_window(text: str, now: datetime = None):
    """
    Parse natural language time → (valid_from, valid_to, horizon).
    Minimal: tomorrow afternoon → 12-18 IST next day.
    Handles: today, tonight, tomorrow, tomorrow afternoon, next 3 days, this weekend, + relative.
    """
    if now is None:
        now = datetime.now(IST)
    text_l = (text or "").lower()
    # defaults
    base = now

    # anchor day
    if "day after tomorrow" in text_l:
        base = now + timedelta(days=2)
    elif "tomorrow" in text_l:
        base = now + timedelta(days=1)
    elif "today" in text_l:
        base = now
    elif "tonight" in text_l:
        base = now
    elif "weekend" in text_l:
        # next saturday
        days_ahead = (5 - now.weekday()) % 7
        base = now + timedelta(days=days_ahead if days_ahead != 0 else 7)

    # time of day windows (IST)
    if "morning" in text_l:
        valid_from = base.replace(hour=6, minute=0, second=0, microsecond=0)
        valid_to = base.replace(hour=11, minute=59, second=59, microsecond=0)
    elif "afternoon" in text_l:
        valid_from = base.replace(hour=12, minute=0, second=0, microsecond=0)
        valid_to = base.replace(hour=18, minute=0, second=0, microsecond=0)
    elif "evening" in text_l:
        valid_from = base.replace(hour=18, minute=0, second=0, microsecond=0)
        valid_to = base.replace(hour=21, minute=0, second=0, microsecond=0)
    elif "night" in text_l or "tonight" in text_l:
        valid_from = base.replace(hour=21, minute=0, second=0, microsecond=0)
        valid_to = (base + timedelta(days=1)).replace(hour=5, minute=59, second=59, microsecond=0)
    elif "next 3 days" in text_l or "next three days" in text_l:
        valid_from = base.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_to = (base + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        # full day
        valid_from = base.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_to = base.replace(hour=23, minute=59, second=59, microsecond=0)

    # horizon
    delta_days = (valid_from.date() - now.date()).days
    if delta_days == 0 and (valid_to - now) <= timedelta(hours=6):
        horizon = "nowcast"
    elif delta_days <= 3:
        horizon = "short"
    elif delta_days <= 10:
        horizon = "medium"
    else:
        horizon = "climate"

    return valid_from, valid_to, horizon

def to_utc_pair(valid_from, valid_to):
    return valid_from.astimezone(timezone.utc), valid_to.astimezone(timezone.utc)
