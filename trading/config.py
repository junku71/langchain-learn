from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from trading.market import get_market_profile


SEOUL = ZoneInfo("Asia/Seoul")
ONEDRIVE_PUBLIC_REPORT_URL = "https://1drv.ms/f/c/714ca3cac310f853/IgCM0r1KM0xSQb1g5kThUnk-AdzbHMoYLJZwWYglmBPg4qM?e=FM3eQu"


def _default_rebalance_report_dir() -> Path:
    onedrive = Path(os.getenv("OneDrive", str(Path.home() / "OneDrive")))
    return onedrive / "Public" / "AI-Stock-Agent"


def _time(value: str) -> time:
    return time.fromisoformat(value.strip())


@dataclass(frozen=True)
class LiveTradingConfig:
    market_region: str = "KR"
    currency: str = "KRW"
    timezone: ZoneInfo = SEOUL
    pre_open_at: time = time(8, 40)
    candidate_snapshot_at: time = time(8, 55)
    market_open_at: time = time(9, 0)
    entry_at: time = time(9, 5)
    market_close_at: time = time(15, 30)
    post_close_at: time = time(15, 40)
    monitor_interval_seconds: int = 60
    ml_filter_enabled: bool = True
    ml_probability_threshold: float = 0.65
    max_candidates_per_market: int = 15
    recommendation_universe_per_market: int = 100
    recommendation_analysis_shortlist: int = 30
    recommendation_final_limit: int = 10
    max_new_positions_per_day: int = 4
    max_daily_orders: int = 20
    max_order_retries: int = 1
    daily_loss_limit_pct: float = 0.03
    quote_stale_seconds: int = 90
    kis_minute_bars_enabled: bool = True
    kis_minute_fallback_to_quote: bool = True
    kis_completed_bars_only: bool = True
    trailing_stop_pct: float = 0.08
    rebalance_enabled: bool = False
    rebalance_llm_model: str = "gpt-5.6"
    rebalance_max_turnover_pct: float = 0.30
    rebalance_max_position_pct: float = 0.20
    rebalance_max_sector_pct: float = 0.40
    rebalance_min_cash_pct: float = 0.10
    rebalance_min_confidence: float = 0.70
    rebalance_proposal_ttl_minutes: int = 10
    rebalance_fill_wait_seconds: int = 30
    rebalance_report_dir: Path = field(default_factory=_default_rebalance_report_dir)
    rebalance_report_base_url: str = ONEDRIVE_PUBLIC_REPORT_URL
    state_db_path: Path = Path("data/trading/live_trading.sqlite3")
    strategy_version: str = "opening-ensemble-v1"
    dry_run: bool = True

    @classmethod
    def from_env(cls) -> "LiveTradingConfig":
        load_dotenv()
        profile = get_market_profile(os.getenv("TRADING_MARKET_REGION", "KR"))
        prefix = "TRADING_US_" if profile.region == "US" else "TRADING_"

        def market_env(name: str, fallback: time) -> time:
            value = os.getenv(f"{prefix}{name}", fallback.isoformat(timespec="minutes"))
            return _time(value)

        return cls(
            market_region=profile.region,
            currency=profile.currency,
            timezone=profile.timezone,
            pre_open_at=market_env("PRE_OPEN_AT", profile.pre_open),
            candidate_snapshot_at=market_env(
                "CANDIDATE_SNAPSHOT_AT", profile.candidate_snapshot
            ),
            market_open_at=market_env("MARKET_OPEN_AT", profile.market_open),
            entry_at=market_env("ENTRY_AT", profile.entry),
            market_close_at=market_env("MARKET_CLOSE_AT", profile.market_close),
            post_close_at=market_env("POST_CLOSE_AT", profile.post_close),
            monitor_interval_seconds=int(
                os.getenv("TRADING_MONITOR_INTERVAL_SECONDS", "60")
            ),
            ml_filter_enabled=os.getenv(
                "TRADING_ML_FILTER_ENABLED", "true"
            ).strip().lower() in {"1", "true", "yes", "on"},
            ml_probability_threshold=float(
                os.getenv("TRADING_ML_PROBABILITY_THRESHOLD", "0.65")
            ),
            max_candidates_per_market=int(
                os.getenv("TRADING_MAX_CANDIDATES_PER_MARKET", "15")
            ),
            recommendation_universe_per_market=int(
                os.getenv("TRADING_RECOMMENDATION_UNIVERSE_PER_MARKET", "100")
            ),
            recommendation_analysis_shortlist=int(
                os.getenv("TRADING_RECOMMENDATION_ANALYSIS_SHORTLIST", "30")
            ),
            recommendation_final_limit=int(
                os.getenv("TRADING_RECOMMENDATION_FINAL_LIMIT", "10")
            ),
            max_new_positions_per_day=int(
                os.getenv("TRADING_MAX_NEW_POSITIONS_PER_DAY", "4")
            ),
            max_daily_orders=int(os.getenv("TRADING_MAX_DAILY_ORDERS", "20")),
            max_order_retries=int(os.getenv("TRADING_MAX_ORDER_RETRIES", "1")),
            daily_loss_limit_pct=float(
                os.getenv("TRADING_DAILY_LOSS_LIMIT_PCT", "0.03")
            ),
            quote_stale_seconds=int(
                os.getenv("TRADING_QUOTE_STALE_SECONDS", "90")
            ),
            kis_minute_bars_enabled=os.getenv(
                "TRADING_KIS_MINUTE_BARS_ENABLED", "true"
            ).strip().lower() in {"1", "true", "yes", "on"},
            kis_minute_fallback_to_quote=os.getenv(
                "TRADING_KIS_MINUTE_FALLBACK_TO_QUOTE", "true"
            ).strip().lower() in {"1", "true", "yes", "on"},
            kis_completed_bars_only=os.getenv(
                "TRADING_KIS_COMPLETED_BARS_ONLY", "true"
            ).strip().lower() in {"1", "true", "yes", "on"},
            trailing_stop_pct=float(
                os.getenv("TRADING_TRAILING_STOP_PCT", "0.08")
            ),
            rebalance_enabled=os.getenv(
                "TRADING_REBALANCE_ENABLED", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            rebalance_llm_model=os.getenv(
                "TRADING_REBALANCE_LLM_MODEL", "gpt-5.6"
            ),
            rebalance_max_turnover_pct=float(
                os.getenv("TRADING_REBALANCE_MAX_TURNOVER_PCT", "0.30")
            ),
            rebalance_max_position_pct=float(
                os.getenv("TRADING_REBALANCE_MAX_POSITION_PCT", "0.20")
            ),
            rebalance_max_sector_pct=float(
                os.getenv("TRADING_REBALANCE_MAX_SECTOR_PCT", "0.40")
            ),
            rebalance_min_cash_pct=float(
                os.getenv("TRADING_REBALANCE_MIN_CASH_PCT", "0.10")
            ),
            rebalance_min_confidence=float(
                os.getenv("TRADING_REBALANCE_MIN_CONFIDENCE", "0.70")
            ),
            rebalance_proposal_ttl_minutes=int(
                os.getenv("TRADING_REBALANCE_PROPOSAL_TTL_MINUTES", "10")
            ),
            rebalance_fill_wait_seconds=int(
                os.getenv("TRADING_REBALANCE_FILL_WAIT_SECONDS", "30")
            ),
            rebalance_report_dir=Path(
                os.getenv(
                    "TRADING_REBALANCE_REPORT_DIR",
                    str(_default_rebalance_report_dir()),
                )
            ),
            rebalance_report_base_url=os.getenv(
                "TRADING_REBALANCE_REPORT_BASE_URL", ONEDRIVE_PUBLIC_REPORT_URL
            ).strip().rstrip("/"),
            state_db_path=Path(
                os.getenv("TRADING_STATE_DB_PATH", "data/trading/live_trading.sqlite3")
            ),
            strategy_version=os.getenv(
                "TRADING_STRATEGY_VERSION", f"opening-ensemble-{profile.region.lower()}-v1"
            ),
            dry_run=os.getenv("TRADING_DRY_RUN", "true").strip().lower()
            in {"1", "true", "yes", "on"},
        )
