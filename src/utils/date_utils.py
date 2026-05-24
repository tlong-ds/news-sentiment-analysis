"""Shared date parsing and calendar utilities.

Consolidates the scattered date-handling logic from the original
``scrape_news.py`` into a single reusable module.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Iterable


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_iso_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` string into a :class:`date`."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def extract_date(value: str | None) -> date | None:
    """Best-effort extraction of a date from free-form text.

    Supports ISO-8601, ``dd/mm/yyyy``, and ``dd-mm-yyyy`` formats commonly
    used across Vietnamese news sites.
    """
    if not value:
        return None
    patterns = [
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
        (r"(\d{1,2}/\d{1,2}/\d{4})", "%d/%m/%Y"),
        (r"(\d{1,2}-\d{1,2}-\d{4})", "%d-%m-%Y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt).date()
            except ValueError:
                continue
    return None


def date_from_unix_seconds(value: str | int | None) -> date | None:
    """Convert a Unix-epoch seconds string/int to a :class:`date`."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def unix_seconds(day: date, end_of_day: bool = False) -> int:
    """Return Unix-epoch seconds for the start (or end) of *day*."""
    clock = datetime_time(23, 59, 59) if end_of_day else datetime_time(0, 0, 0)
    return int(datetime.combine(day, clock).timestamp())


# ---------------------------------------------------------------------------
# Date-range iteration
# ---------------------------------------------------------------------------

def date_blocks(start: date, end: date, days: int) -> Iterable[tuple[date, date]]:
    """Yield ``(block_start, block_end)`` tuples spanning *start* to *end*."""
    current = start
    while current <= end:
        block_end = min(current + timedelta(days=days - 1), end)
        yield current, block_end
        current = block_end + timedelta(days=1)


def month_ranges(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Yield ``(month_start, month_end)`` tuples spanning *start* to *end*."""
    current = start.replace(day=1)
    while current <= end:
        import calendar
        _, last_day = calendar.monthrange(current.year, current.month)
        month_end = current.replace(day=last_day)
        yield max(current, start), min(month_end, end)
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)
