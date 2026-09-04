from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketProfile:
    region: str
    display_name: str
    timezone: ZoneInfo
    currency: str
    currency_symbol: str
    universes: tuple[str, ...]
    default_scope: str
    market_open: time
    market_close: time
    pre_open: time
    candidate_snapshot: time
    entry: time
    post_close: time


MARKET_PROFILES = {
    "KR": MarketProfile(
        "KR", "KRX", ZoneInfo("Asia/Seoul"), "KRW", "₩",
        ("KOSPI", "KOSDAQ"), "BOTH", time(9), time(15, 30),
        time(8, 40), time(8, 55), time(9, 5), time(15, 40),
    ),
    "US": MarketProfile(
        "US", "NASDAQ / S&P 500", ZoneInfo("America/New_York"), "USD", "$",
        ("NASDAQ", "SP500"), "BOTH", time(9, 30), time(16),
        time(9, 10), time(9, 20), time(9, 35), time(16, 10),
    ),
}


def get_market_profile(region: str) -> MarketProfile:
    key = region.strip().upper()
    if key not in MARKET_PROFILES:
        raise ValueError("TRADING_MARKET_REGION must be KR or US")
    return MARKET_PROFILES[key]
