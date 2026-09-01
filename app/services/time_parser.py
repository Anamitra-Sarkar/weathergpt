from __future__ import annotations
from datetime import datetime, timedelta, timezone
import re

IST = timezone(timedelta(hours=5, minutes=30))

def parse_time_window(text: str, now: datetime = None, tz: timezone = IST):
    """
    Deterministic time normalization → (valid_from, valid_to, horizon, resolution_confidence).
    Supports: today, tonight, tomorrow, tomorrow morning, day after tomorrow, next 3 days, next week, this weekend, coming Monday, explicit dates, relative hours.
    Returns IST-aware datetimes, horizon, and confidence.
    """
    if now is None:
        now = datetime.now(tz)
    text_l = (text or "").lower()
    base = now
    confidence = 0.9

    # explicit date like 2024-01-15 or 23rd Aug
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text_l)
    if m:
        try:
            base = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=tz)
            confidence = 1.0
        except:
            confidence = 0.5
    elif re.search(r"(\d{1,2})(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text_l):
        # e.g., 23rd Aug -> assume current year
        try:
            import calendar
            # crude parse
            for i, mon in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],1):
                if mon in text_l:
                    d = int(re.search(r"(\d{1,2})", text_l).group(1))
                    base = datetime(now.year, i, d, tzinfo=tz)
                    if base < now:
                        base = base.replace(year=now.year+1)
                    confidence = 0.8
                    break
        except:
            pass
    elif "day after tomorrow" in text_l or "parso" in text_l or "परसों" in text_l:
        base = now + timedelta(days=2)
    elif "tomorrow" in text_l or "kal" in text_l or "कल" in text_l:
        base = now + timedelta(days=1)
    elif "today" in text_l:
        base = now
    elif "tonight" in text_l:
        base = now
    elif "next week" in text_l:
        base = now + timedelta(days=7)
        confidence = 0.7
    elif "coming monday" in text_l or "next monday" in text_l:
        days_ahead = (0 - now.weekday()) % 7
        if days_ahead == 0: days_ahead = 7
        base = now + timedelta(days=days_ahead)
    elif "weekend" in text_l:
        days_ahead = (5 - now.weekday()) % 7
        base = now + timedelta(days=days_ahead if days_ahead != 0 else 7)
    elif "next 3 days" in text_l or "next three days" in text_l:
        pass  # handled below
    elif re.search(r"in (\d+) hours?", text_l):
        hrs = int(re.search(r"in (\d+) hours?", text_l).group(1))
        base = now + timedelta(hours=hrs)
        return base, base + timedelta(hours=1), "nowcast", 0.95

    # time of day windows
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
        confidence = 0.8
    elif "next week" in text_l:
        valid_from = base.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_to = (base + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=0)
        confidence = 0.7
    else:
        valid_from = base.replace(hour=0, minute=0, second=0, microsecond=0)
        valid_to = base.replace(hour=23, minute=59, second=59, microsecond=0)
        if base == now and "tomorrow" not in text_l and "today" not in text_l:
            confidence = 0.6

    delta_days = (valid_from.date() - now.date()).days
    if delta_days == 0 and (valid_to - now) <= timedelta(hours=6):
        horizon = "nowcast"
    elif delta_days <= 3:
        horizon = "short"
    elif delta_days <= 10:
        horizon = "medium"
    else:
        horizon = "climate"

    return valid_from, valid_to, horizon, confidence

def to_utc_pair(valid_from, valid_to):
    return valid_from.astimezone(timezone.utc), valid_to.astimezone(timezone.utc)
