"""Inclusive calendar filtering for match history rows."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def played_on_from_value(value: Any) -> date | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value).date()
    except ValueError:
        try:
            return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S.%f").date()
        except ValueError:
            return None


def in_inclusive_date_range(
    played_on: date | None,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Keep a match when its calendar day is inside an inclusive range.

    Rows with no parseable date are hidden once either bound is set, and kept
    when the user has not applied a date filter.
    """
    if start_date is not None and (played_on is None or played_on < start_date):
        return False
    if end_date is not None and (played_on is None or played_on > end_date):
        return False
    return True
