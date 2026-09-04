from __future__ import annotations

from broker.trading_context import broker, trade_logger
from broker.kis import KISBroker
from trading.calendar import KrxCalendar, UsMarketCalendar
from trading.config import LiveTradingConfig
from trading.graphs import TradingGraphContext
from trading.notifications import NullNotifier, SlackConfig, SlackNotifier
from trading.providers import (
    BrokerQuoteProvider,
    KISMinuteBarProvider,
    ModelCandidateProvider,
    PassthroughCandidateAnalysis,
)
from trading.service import LiveTradingService
from trading.reconciliation import OrderReconciler
from trading.state_store import TradingStateStore


def create_live_trading_service(
    config: LiveTradingConfig | None = None,
) -> LiveTradingService:
    config = config or LiveTradingConfig.from_env()
    calendar_type = UsMarketCalendar if config.market_region == "US" else KrxCalendar
    calendar = calendar_type(
        timezone=config.timezone,
        default_open=config.market_open_at,
        default_close=config.market_close_at,
    )
    quote_provider = BrokerQuoteProvider(broker)
    if config.kis_minute_bars_enabled and isinstance(broker, KISBroker):
        quote_provider = KISMinuteBarProvider(
            broker,
            fallback_to_current_price=config.kis_minute_fallback_to_quote,
            completed_bars_only=config.kis_completed_bars_only,
        )
    context = TradingGraphContext(
        config=config,
        calendar=calendar,
        store=TradingStateStore(config.state_db_path),
        broker=broker,
        quote_provider=quote_provider,
        candidate_provider=ModelCandidateProvider(),
        analysis=PassthroughCandidateAnalysis(),
        trade_logger=trade_logger,
    )
    slack_config = SlackConfig.from_env()
    notifier = SlackNotifier(slack_config) if slack_config.enabled else NullNotifier()
    reconciler = OrderReconciler(broker, context.store, notifier)
    return LiveTradingService(context, reconciler=reconciler)
