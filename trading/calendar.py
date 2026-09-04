from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from trading.config import SEOUL


@dataclass(frozen=True)
class MarketSession:
    trade_date: date
    opens_at: datetime
    closes_at: datetime

    def is_open(self, now: datetime) -> bool:
        return self.opens_at <= now < self.closes_at


@dataclass
class KrxCalendar:
    """KRX session calendar with injectable holidays and special hours.

    Production should populate holidays/special_sessions from an authoritative
    exchange calendar before the service starts. Weekends are always excluded.
    """

    timezone: ZoneInfo = SEOUL
    holidays: set[date] = field(default_factory=set)
    special_sessions: dict[date, tuple[time, time]] = field(default_factory=dict)
    default_open: time = time(9, 0)
    default_close: time = time(15, 30)

    def is_trading_day(self, value: date) -> bool:
        return value.weekday() < 5 and value not in self.holidays

    def next_trading_day(self, value: date, *, include_today: bool = False) -> date:
        candidate = value if include_today else value + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def session(self, value: date) -> MarketSession | None:
        if not self.is_trading_day(value):
            return None
        opens, closes = self.special_sessions.get(
            value, (self.default_open, self.default_close)
        )
        return MarketSession(
            trade_date=value,
            opens_at=datetime.combine(value, opens, self.timezone),
            closes_at=datetime.combine(value, closes, self.timezone),
        )

    def phase(self, now: datetime) -> str:
        localized = now.astimezone(self.timezone)
        session = self.session(localized.date())
        if session is None:
            return "CLOSED_DAY"
        if localized < session.opens_at:
            return "PRE_OPEN"
        if session.is_open(localized):
            return "REGULAR"
        return "POST_CLOSE"


@dataclass
class UsMarketCalendar(KrxCalendar):
    """US regular session calendar in America/New_York.

    Holiday and early-close dates remain injectable, just like KrxCalendar.
    Populate them from the broker/exchange calendar in production.
    """

    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("America/New_York"))
    default_open: time = time(9, 30)
    default_close: time = time(16, 0)
