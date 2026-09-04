from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from analysis.ticker_mapper import get_company_name


SEOUL = ZoneInfo("Asia/Seoul")


def korean_time(value: object) -> str:
    """Format a stored ISO timestamp as Korea Standard Time."""
    if value in (None, ""):
        return "-"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(SEOUL).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def market_time(value: object, timezone_name: str = "Asia/Seoul") -> str:
    """Format a stored ISO timestamp in a market timezone."""
    if value in (None, ""):
        return "-"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def money(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


@lru_cache(maxsize=4096)
def stock_name(ticker: str) -> str:
    """Resolve a display name from the cached symbol masters."""
    code = ticker.split(".", 1)[0].upper()
    name = get_company_name(ticker)
    if name != code:
        return name
    # KIS balance currently normalizes domestic positions with a .KS suffix.
    # Fall back to both markets so KOSDAQ holdings still receive their names.
    for market in ("KOSPI", "KOSDAQ", "NASDAQ", "SP500"):
        name = get_company_name(code, market)
        if name != code:
            return name
    return code


def table(headers: list[str], rows: list[list[object]]) -> str:
    def width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in value
        )

    text_rows = [["-" if value is None else str(value) for value in row] for row in rows]
    if not text_rows:
        return "(데이터 없음)"
    widths = [
        max(width(header), *(width(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: list[str]) -> str:
        return " | ".join(
            value + " " * (widths[index] - width(value))
            for index, value in enumerate(row)
        )

    return "\n".join(
        [render(headers), "-+-".join("-" * item for item in widths)]
        + [render(row) for row in text_rows]
    )
