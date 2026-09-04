from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from trading.calendar import UsMarketCalendar
from trading.config import LiveTradingConfig
from trading.market import get_market_profile


def test_us_profile_uses_eastern_time_and_us_regular_session():
    profile = get_market_profile("US")
    assert profile.currency == "USD"
    assert profile.universes == ("NASDAQ", "SP500")
    assert profile.market_open == time(9, 30)
    assert str(profile.timezone) == "America/New_York"


def test_us_calendar_handles_dst_through_zoneinfo():
    calendar = UsMarketCalendar()
    winter = calendar.session(date(2026, 1, 5))
    summer = calendar.session(date(2026, 7, 6))
    assert winter.opens_at.utcoffset().total_seconds() == -5 * 3600
    assert summer.opens_at.utcoffset().total_seconds() == -4 * 3600
    assert calendar.phase(datetime(2026, 7, 6, 10, tzinfo=ZoneInfo("America/New_York"))) == "REGULAR"


def test_us_config_ignores_legacy_krx_time_variables(monkeypatch):
    monkeypatch.setenv("TRADING_MARKET_REGION", "US")
    monkeypatch.setenv("TRADING_MARKET_OPEN_AT", "09:00")
    monkeypatch.delenv("TRADING_US_MARKET_OPEN_AT", raising=False)
    config = LiveTradingConfig.from_env()
    assert config.market_region == "US"
    assert config.currency == "USD"
    assert config.market_open_at == time(9, 30)


def test_invalid_market_region_is_rejected():
    with pytest.raises(ValueError):
        get_market_profile("EU")
