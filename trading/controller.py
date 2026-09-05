from __future__ import annotations

import threading
import time
import uuid
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from broker.models import OrderResult
from analysis.ticker_mapper import get_domestic_security, get_security
from trading.market import get_market_profile
from portfolio_manager import PortfolioLimits, can_add_position
from trading.display import stock_name
from trading.graphs import TradingGraphContext
from trading.models import ProtectionState
from trading.recommendations import RecommendationService
from trading.recommendation_report import render_top10_pick_report
from trading.rebalance_report import render_rebalance_report
from trading.rebalance import (
    LLMRebalanceAdvisor,
    MarketNewsService,
    RebalanceExecutor,
    RebalanceProposal,
    RebalanceValidator,
    proposal_id,
)
from trading.service import JobResult, LiveTradingService
from trading.trade_logging import log_trade_safely


class TradingControlError(RuntimeError):
    pass


class TradingController:
    """Safe application layer shared by the interactive console and tests."""

    def __init__(
        self,
        service: LiveTradingService,
        recommendation_service: RecommendationService | None = None,
        atr_provider: Callable[[str], float] | None = None,
        rebalance_advisor=None,
        market_news_service=None,
    ):
        self.service = service
        self.context: TradingGraphContext = service.context
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_stop = threading.Event()
        self._scheduler_last_result: JobResult | None = None
        self._scheduler_error: str | None = None
        self.recommendation_service = recommendation_service or RecommendationService(
            self.context
        )
        self.atr_provider = atr_provider or self._latest_daily_atr
        self.rebalance_advisor = rebalance_advisor
        self.market_news_service = market_news_service or MarketNewsService()

    @staticmethod
    def _latest_daily_atr(ticker: str) -> float:
        from analysis.technical import calculate_indicators, get_stock_data

        frame = calculate_indicators(get_stock_data(ticker, period="1y"))
        atr = float(frame.iloc[-1]["ATR"])
        if atr <= 0:
            raise ValueError("ATR is unavailable or non-positive")
        return atr

    def now(self) -> datetime:
        return datetime.now(self.context.config.timezone)

    def environment(self) -> dict:
        broker = self.context.broker
        broker_config = getattr(broker, "config", None)
        account_type = getattr(broker_config, "account_type", "PAPER")
        enabled = getattr(broker_config, "enable_trading", True)
        session = self.current_session()
        return {
            "market_region": self.context.config.market_region,
            "currency": self.context.config.currency,
            "broker": type(broker).__name__,
            "account_type": account_type,
            "trading_enabled": bool(enabled),
            "dry_run": self.context.config.dry_run,
            "market_phase": self.context.calendar.phase(self.now()),
            "scheduler": "RUNNING" if self.scheduler_running else "STOPPED",
            "kill_switch": (session or {}).get("kill_switch", "NORMAL"),
            "buy_enabled": True,
            "ml_filter": "ON" if self.ml_filter_enabled() else "OFF",
            "slack": (
                "ON"
                if self.service.reconciler
                and self.service.reconciler.notifier.enabled
                else "OFF"
            ),
        }

    def ml_filter_enabled(self) -> bool:
        if self.context.config.market_region == "US":
            return False
        return True

    def toggle_ml_filter(self) -> bool:
        if self.context.config.market_region == "US":
            raise TradingControlError(
                "미국시장용 ML 모델이 아직 없으므로 US 모드에서는 ML Filter를 사용할 수 없습니다."
            )
        self.context.store.set_control("ml_filter_enabled", True)
        self.context.store.audit(
            "ML_FILTER_CHANGED", "ml_filter_enabled", {"enabled": True}
        )
        return True

    def account_snapshot(self) -> tuple[dict, dict]:
        positions = self.context.broker.get_positions()
        balance = self.context.broker.get_balance()
        return balance, positions

    def current_session_id(self) -> str:
        suffix = ":US" if self.context.config.market_region == "US" else ""
        return f"{self.now().date().isoformat()}:{self.context.config.strategy_version}{suffix}"

    def current_session(self) -> dict | None:
        return self.context.store.get_session(self.current_session_id())

    def _ensure_manual_session(self) -> dict:
        session = self.current_session()
        if session is not None:
            return session
        now = self.now()
        session_id = self.current_session_id()
        self.context.store.upsert_session(
            session_id,
            now.date().isoformat(),
            self.context.config.strategy_version,
            "MANUAL_READY",
            buy_enabled=True,
            payload={"created_by": "console", "created_at": now.isoformat()},
        )
        return self.context.store.get_session(session_id) or {}

    def quote(self, ticker: str) -> float:
        return float(self.context.broker.get_current_price(ticker))

    def _validate_order_window(self) -> None:
        if self.context.calendar.phase(self.now()) != "REGULAR":
            raise TradingControlError("정규장 운영시간에만 수동 주문할 수 있습니다.")

    def _record_order(self, key: str, result: OrderResult) -> None:
        payload = asdict(result)
        self.context.store.update_order_intent(
            key,
            result.status,
            broker_order_id=result.order_id,
            payload=payload,
        )
        self.context.store.audit("MANUAL_ORDER", key, payload)
        log_trade_safely(self.context, result, entity_key=key)

    def manual_buy(
        self,
        ticker: str,
        quantity: int,
        *,
        limit_price: float | None = None,
        sector: str = "UNKNOWN",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
        order_type: str = "LIMIT",
    ) -> OrderResult:
        self._validate_order_window()
        if quantity <= 0:
            raise TradingControlError("매수 수량은 1주 이상이어야 합니다.")
        normalized_order_type = order_type.strip().upper()
        if normalized_order_type not in {
            "LIMIT", "MARKET", "PRIORITY_LIMIT", "BEST_LIMIT"
        }:
            raise TradingControlError("지원하지 않는 매수 가격 방식입니다.")
        price = float(limit_price or self.quote(ticker))
        if price <= 0:
            raise TradingControlError("주문 가격이 올바르지 않습니다.")
        if stop_loss is not None and stop_loss >= price:
            raise TradingControlError("손절가는 주문 가격보다 낮아야 합니다.")
        if take_profit is not None and take_profit <= price:
            raise TradingControlError("익절가는 주문 가격보다 높아야 합니다.")
        if trailing_stop_pct is not None and not 0 < trailing_stop_pct < 1:
            raise TradingControlError("Trailing stop은 0% 초과 100% 미만이어야 합니다.")

        session = self._ensure_manual_session()
        if session.get("kill_switch") == "HALTED":
            raise TradingControlError("Kill Switch가 활성화되어 있습니다.")
        session_id = self.current_session_id()
        if self.context.store.count_orders(session_id) >= self.context.config.max_daily_orders:
            raise TradingControlError("일일 주문 횟수 한도에 도달했습니다.")

        positions = self.context.broker.get_positions()
        prices = {}
        for held_ticker, position in positions.items():
            try:
                prices[held_ticker] = self.quote(held_ticker)
            except Exception:
                prices[held_ticker] = position.avg_price
        report = self.context.portfolio_manager.evaluate(prices)
        guard = can_add_position(
            report,
            ticker=ticker,
            sector=sector,
            new_position_value=price * quantity,
            limits=PortfolioLimits(),
        )
        if not guard.get("approved"):
            raise TradingControlError(f"포트폴리오 제한: {guard.get('reason')}")

        key = f"{session_id}:MANUAL:{uuid.uuid4().hex}:BUY:{ticker}"
        intent = {
            "price": price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop_pct": trailing_stop_pct,
            "order_type": normalized_order_type,
            "source": "console",
        }
        if not self.context.store.create_order_intent(
            key, session_id, ticker, "BUY", intent
        ):
            raise TradingControlError("중복 주문 의도가 감지되었습니다.")

        if self.context.config.dry_run:
            result = OrderResult(
                status="DRY_RUN", ticker=ticker, side="BUY", price=price,
                quantity=quantity, reason="USER_MANUAL_BUY",
            )
        else:
            result = self.context.broker.buy(
                ticker=ticker,
                price=price,
                quantity=quantity,
                sector=sector,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
                order_type=normalized_order_type,
                reason="USER_MANUAL_BUY",
            )
        self._record_order(key, result)
        if result.status in {"FILLED", "SUBMITTED"}:
            self.context.store.save_protection(
                ProtectionState(
                    ticker=ticker,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_pct=trailing_stop_pct,
                    trailing_stop=(
                        price * (1 - trailing_stop_pct)
                        if trailing_stop_pct is not None else None
                    ),
                    highest_price=price,
                    updated_at=self.now(),
                )
            )
        return result

    def resolve_domestic_security(self, company_name: str) -> dict:
        profile = get_market_profile(self.context.config.market_region)
        try:
            security = (
                get_domestic_security(company_name)
                if profile.region == "KR"
                else get_security(company_name, profile.universes)
            )
        except (KeyError, ValueError, OSError) as error:
            raise TradingControlError(
                f"종목명 '{company_name}'을(를) KOSPI/KOSDAQ에서 찾을 수 없습니다: {error}"
            ) from error
        sector = "UNKNOWN"
        universe_path = getattr(self.recommendation_service, "universe_path", None)
        if universe_path and Path(universe_path).exists():
            with Path(universe_path).open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    if str(row.get("ticker") or "").upper() == security["ticker"].upper():
                        sector = str(row.get("sector") or "UNKNOWN")
                        break
        return {**security, "sector": sector}

    def resolve_security(self, company_name: str) -> dict:
        return self.resolve_domestic_security(company_name)

    def manual_buy_or_reserve(
        self,
        ticker: str,
        quantity: int,
        *,
        limit_price: float,
        sector: str,
        name: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
        order_type: str = "LIMIT",
    ):
        if self.context.calendar.phase(self.now()) == "REGULAR":
            return self.manual_buy(
                ticker, quantity, limit_price=limit_price, sector=sector,
                stop_loss=stop_loss, take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct, order_type=order_type,
            )
        normalized_order_type = order_type.strip().upper()
        if normalized_order_type not in {
            "LIMIT", "MARKET", "PRIORITY_LIMIT", "BEST_LIMIT"
        }:
            raise TradingControlError("지원하지 않는 매수 가격 방식입니다.")
        if quantity <= 0 or limit_price <= 0:
            raise TradingControlError("예약 매수 수량과 지정가가 올바르지 않습니다.")
        if stop_loss is not None and stop_loss >= limit_price:
            raise TradingControlError("손절가는 지정가보다 낮아야 합니다.")
        if take_profit is not None and take_profit <= limit_price:
            raise TradingControlError("익절가는 지정가보다 높아야 합니다.")
        if trailing_stop_pct is not None and not 0 < trailing_stop_pct < 1:
            raise TradingControlError("Trailing stop은 0% 초과 100% 미만이어야 합니다.")
        session = self._ensure_manual_session()
        if session.get("kill_switch") == "HALTED":
            raise TradingControlError("Kill Switch가 활성화되어 있습니다.")
        now = self.now().astimezone(self.context.config.timezone)
        phase = self.context.calendar.phase(now)
        execute_date = self.context.calendar.next_trading_day(
            now.date(), include_today=phase == "PRE_OPEN"
        )
        reservation_id = f"MANUAL-BUY:{uuid.uuid4().hex}:{ticker}"
        payload = {
            "name": name, "sector": sector,
            "limit_price": limit_price if normalized_order_type == "LIMIT" else None,
            "reference_price": limit_price,
            "order_type": normalized_order_type,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "trailing_stop_pct": trailing_stop_pct,
            "approved_at": now.isoformat(), "source": "MANUAL_BUY",
        }
        if not self.context.store.enqueue_scheduled_order(
            reservation_id, "MANUAL_BUY", execute_date.isoformat(), ticker,
            "BUY", quantity, payload,
        ):
            raise TradingControlError("예약 매수 주문을 저장하지 못했습니다.")
        result = {
            "status": "QUEUED", "reservation_id": reservation_id,
            "execute_on": execute_date.isoformat(), "ticker": ticker,
            "side": "BUY", "quantity": quantity, "price": limit_price,
            **payload,
        }
        self.context.store.audit("MANUAL_BUY_RESERVED", reservation_id, result)
        return result

    def manual_sell(
        self,
        ticker: str,
        quantity: int,
        *,
        limit_price: float | None = None,
        order_type: str = "LIMIT",
    ) -> OrderResult:
        self._validate_order_window()
        position = self.context.broker.get_position(ticker)
        if position is None:
            raise TradingControlError("보유하지 않은 종목입니다.")
        if quantity <= 0 or quantity > position.quantity:
            raise TradingControlError("매도 수량이 보유수량 범위를 벗어났습니다.")
        normalized_order_type = order_type.strip().upper()
        if normalized_order_type not in {
            "LIMIT", "MARKET", "PRIORITY_LIMIT", "BEST_LIMIT"
        }:
            raise TradingControlError("지원하지 않는 매도 가격 방식입니다.")
        price = float(limit_price or self.quote(ticker))
        session = self._ensure_manual_session()
        if session.get("kill_switch") == "HALTED":
            raise TradingControlError("Kill Switch가 활성화되어 있습니다.")
        session_id = self.current_session_id()
        key = f"{session_id}:MANUAL:{uuid.uuid4().hex}:SELL:{ticker}"
        intent = {
            "price": price, "quantity": quantity, "source": "console",
            "order_type": normalized_order_type,
        }
        self.context.store.create_order_intent(key, session_id, ticker, "SELL", intent)
        if self.context.config.dry_run:
            result = OrderResult(
                status="DRY_RUN", ticker=ticker, side="SELL", price=price,
                quantity=quantity, reason="USER_MANUAL_SELL",
            )
        else:
            result = self.context.broker.sell(
                ticker=ticker, price=price, quantity=quantity,
                order_type=normalized_order_type,
                reason="USER_MANUAL_SELL",
            )
        self._record_order(key, result)
        if result.status == "FILLED" and quantity == position.quantity:
            self.context.store.delete_protection(ticker)
        return result

    def set_protection(
        self,
        ticker: str,
        *,
        stop_loss: float | None,
        take_profit: float | None,
        trailing_stop_pct: float | None,
    ) -> ProtectionState:
        position = self.context.broker.get_position(ticker)
        if position is None:
            raise TradingControlError("보유하지 않은 종목입니다.")
        current = self.quote(ticker)
        if stop_loss is not None and stop_loss >= current:
            raise TradingControlError("손절가는 현재가보다 낮아야 합니다.")
        if take_profit is not None and take_profit <= current:
            raise TradingControlError("익절가는 현재가보다 높아야 합니다.")
        if trailing_stop_pct is not None and not 0 < trailing_stop_pct < 1:
            raise TradingControlError("Trailing stop 범위가 올바르지 않습니다.")
        previous = self.context.store.get_protection(ticker)
        highest = max(current, previous.highest_price if previous else position.avg_price)
        state = ProtectionState(
            ticker=ticker,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop=(highest * (1 - trailing_stop_pct) if trailing_stop_pct else None),
            highest_price=highest,
            updated_at=self.now(),
        )
        self.context.store.save_protection(state)
        self.context.store.audit("PROTECTION_UPDATED", ticker, asdict(state))
        return state

    @staticmethod
    def _latest_donchian_low(ticker: str, period: int = 20) -> float:
        from trading.protection import latest_donchian_low

        return latest_donchian_low(ticker, period)

    def preview_exit_strategy(
        self,
        ticker: str,
        strategy: str,
        *,
        atr_multiple: float = 3.0,
        donchian_period: int = 20,
        direct_take_profit_pct: float = 0.20,
        direct_stop_loss_pct: float = 0.10,
        direct_trailing_stop_pct: float = 0.08,
    ) -> dict:
        position = self.context.broker.get_position(ticker)
        if position is None:
            raise TradingControlError("보유하지 않은 종목입니다.")
        normalized = strategy.strip().upper()
        if normalized not in {
            "CHANDELIER_EXIT", "DONCHIAN_TREND", "DIRECT_SPECIFIED"
        }:
            raise TradingControlError("지원하지 않는 손절 전략입니다.")
        if atr_multiple <= 0:
            raise TradingControlError("ATR 배수는 0보다 커야 합니다.")
        if donchian_period < 2:
            raise TradingControlError("돈치안 기간은 2일 이상이어야 합니다.")
        if normalized == "DIRECT_SPECIFIED" and not (
            direct_take_profit_pct > 0
            and 0 < direct_stop_loss_pct < 1
            and 0 < direct_trailing_stop_pct < 1
        ):
            raise TradingControlError("직접지정 비율이 올바르지 않습니다.")

        current = self.quote(ticker)
        atr = float(self.atr_provider(ticker))
        if atr <= 0:
            raise TradingControlError("ATR을 계산할 수 없습니다.")
        previous = self.context.store.get_protection(ticker)
        highest = max(
            current, position.avg_price,
            previous.highest_price if previous is not None else 0,
        )
        if normalized == "CHANDELIER_EXIT":
            initial_stop = max(0.0, position.avg_price - atr_multiple * atr)
            trailing_stop = max(initial_stop, highest - atr_multiple * atr)
            label = "샹들리에 Exit"
            period = None
        elif normalized == "DONCHIAN_TREND":
            channel_low = self._latest_donchian_low(ticker, donchian_period)
            initial_stop = channel_low
            trailing_stop = channel_low
            label = f"돈치안 추세추종({donchian_period}일)"
            period = donchian_period
        else:
            initial_stop = position.avg_price * (1 - direct_stop_loss_pct)
            take_profit = position.avg_price * (1 + direct_take_profit_pct)
            trailing_stop = highest * (1 - direct_trailing_stop_pct)
            label = (
                f"직접지정(익절 +{direct_take_profit_pct:.0%}, "
                f"손절 -{direct_stop_loss_pct:.0%}, "
                f"Trailing {direct_trailing_stop_pct:.0%})"
            )
            period = None
        return {
            "ticker": ticker, "name": stock_name(ticker),
            "strategy": normalized, "strategy_name": label,
            "avg_price": position.avg_price, "current_price": current,
            "atr": atr, "atr_multiple": atr_multiple,
            "stop_loss": initial_stop,
            "take_profit": take_profit if normalized == "DIRECT_SPECIFIED" else None,
            "trailing_stop_pct": (
                direct_trailing_stop_pct if normalized == "DIRECT_SPECIFIED" else None
            ),
            "trailing_stop": trailing_stop,
            "highest_price": highest, "donchian_period": period,
            "warning": "현재가가 손절 기준 이하입니다" if current <= trailing_stop else "",
        }

    def apply_exit_strategy(self, preview: dict) -> ProtectionState:
        state = ProtectionState(
            ticker=preview["ticker"], stop_loss=preview["stop_loss"],
            take_profit=preview.get("take_profit"),
            trailing_stop_pct=preview.get("trailing_stop_pct"),
            trailing_stop=preview["trailing_stop"],
            highest_price=preview["highest_price"], updated_at=self.now(),
            strategy=preview["strategy"], atr=preview["atr"],
            atr_multiple=preview["atr_multiple"],
            donchian_period=preview.get("donchian_period"),
        )
        self.context.store.save_protection(state)
        self.context.store.audit("EXIT_STRATEGY_UPDATED", state.ticker, asdict(state))
        return state

    def exit_strategy_overview(self) -> list[dict]:
        """Return every holding's exit setup, initializing missing strategies."""
        positions = self.context.broker.get_positions()
        if not positions:
            raise TradingControlError("보유종목이 없습니다.")
        rows = []
        labels = {
            "CHANDELIER_EXIT": "샹들리에 Exit",
            "DONCHIAN_TREND": "돈치안 추세추종",
            "DIRECT_SPECIFIED": "직접지정전략",
        }
        for ticker, position in positions.items():
            try:
                current_price = float(self.quote(ticker))
                protection = self.context.store.get_protection(ticker)
                if protection is None or protection.strategy == "LEGACY":
                    preview = self.preview_exit_strategy(
                        ticker, "CHANDELIER_EXIT", atr_multiple=3.0
                    )
                    protection = self.apply_exit_strategy(preview)
                atr = protection.atr
                if atr is None:
                    atr = float(self.atr_provider(ticker))
                rows.append({
                    "ticker": ticker, "name": stock_name(ticker),
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": current_price,
                    "return_pct": (
                        (current_price - position.avg_price) / position.avg_price
                        if position.avg_price > 0 else None
                    ),
                    "total_value": position.avg_price * position.quantity,
                    "strategy": protection.strategy,
                    "strategy_name": labels.get(protection.strategy, protection.strategy),
                    "atr": atr, "take_profit": protection.take_profit,
                    "stop_loss": protection.stop_loss,
                    "trailing_stop": protection.trailing_stop,
                    "status": "READY", "message": "",
                })
            except Exception as error:
                rows.append({
                    "ticker": ticker, "name": stock_name(ticker),
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": None,
                    "return_pct": None,
                    "total_value": position.avg_price * position.quantity,
                    "strategy": "", "strategy_name": "계산 실패",
                    "atr": None, "take_profit": None, "stop_loss": None,
                    "trailing_stop": None, "status": "ERROR",
                    "message": f"{type(error).__name__}: {error}",
                })
        return rows

    def preview_bulk_exit_strategy(
        self,
        strategy: str,
        *,
        tickers: list[str] | None = None,
        atr_multiple: float = 3.0,
        donchian_period: int = 20,
        direct_take_profit_pct: float = 0.20,
        direct_stop_loss_pct: float = 0.10,
        direct_trailing_stop_pct: float = 0.08,
    ) -> list[dict]:
        positions = self.context.broker.get_positions()
        if not positions:
            raise TradingControlError("보유종목이 없습니다.")
        selected = list(positions) if tickers is None else [
            str(ticker).upper() for ticker in tickers
        ]
        unknown = [ticker for ticker in selected if ticker not in positions]
        if unknown:
            raise TradingControlError(
                f"보유하지 않은 종목이 포함되어 있습니다: {', '.join(unknown)}"
            )
        results = []
        for ticker in selected:
            try:
                preview = self.preview_exit_strategy(
                    ticker, strategy, atr_multiple=atr_multiple,
                    donchian_period=donchian_period,
                    direct_take_profit_pct=direct_take_profit_pct,
                    direct_stop_loss_pct=direct_stop_loss_pct,
                    direct_trailing_stop_pct=direct_trailing_stop_pct,
                )
                preview.update({"status": "READY", "message": preview["warning"]})
            except Exception as error:
                preview = {
                    "ticker": ticker, "name": stock_name(ticker),
                    "strategy": strategy, "strategy_name": strategy,
                    "atr": None, "stop_loss": None, "take_profit": None,
                    "trailing_stop": None, "status": "ERROR",
                    "message": f"{type(error).__name__}: {error}",
                }
            results.append(preview)
        return results

    def apply_bulk_exit_strategy(self, previews: list[dict]) -> list[dict]:
        results = []
        for preview in previews:
            if preview.get("status") == "ERROR":
                results.append(dict(preview))
                continue
            state = self.apply_exit_strategy(preview)
            results.append({**preview, "status": "SAVED", "state": asdict(state)})
        return results

    def preview_bulk_atr_protection(
        self,
        trailing_stop_pct: float | None = None,
    ) -> list[dict]:
        trailing_pct = (
            self.context.config.trailing_stop_pct
            if trailing_stop_pct is None else trailing_stop_pct
        )
        if not 0 < trailing_pct < 1:
            raise TradingControlError(
                "Trailing stop 비율은 0%보다 크고 100%보다 작아야 합니다."
            )
        positions = self.context.broker.get_positions()
        if not positions:
            raise TradingControlError("보유종목이 없습니다.")
        results = []
        for ticker, position in positions.items():
            try:
                current = self.quote(ticker)
                atr = float(self.atr_provider(ticker))
                if atr <= 0:
                    raise ValueError("ATR must be positive")
                stop_loss = max(0.0, position.avg_price - 3 * atr)
                take_profit = position.avg_price * 1.20
                previous = self.context.store.get_protection(ticker)
                highest = max(
                    current,
                    position.avg_price,
                    previous.highest_price if previous is not None else 0,
                )
                trailing_stop = (
                    max(
                        (previous.trailing_stop if previous is not None else 0) or 0,
                        highest * (1 - trailing_pct),
                    )
                    if trailing_pct is not None else None
                )
                warnings = []
                if stop_loss >= current:
                    warnings.append("즉시 손절 가능")
                if take_profit <= current:
                    warnings.append("즉시 익절 가능")
                result = {
                    "ticker": ticker,
                    "avg_price": position.avg_price,
                    "current_price": current,
                    "atr": atr,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop_pct": trailing_pct,
                    "trailing_stop": trailing_stop,
                    "highest_price": highest,
                    "status": "WARNING" if warnings else "READY",
                    "message": ", ".join(warnings) if warnings else "적용 가능",
                }
            except Exception as error:
                result = {
                    "ticker": ticker,
                    "avg_price": position.avg_price,
                    "current_price": None,
                    "atr": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "trailing_stop_pct": None,
                    "trailing_stop": None,
                    "highest_price": position.avg_price,
                    "status": "ERROR",
                    "message": f"{type(error).__name__}: {error}",
                }
            results.append(result)
        return results

    def apply_bulk_atr_protection(self, preview: list[dict]) -> list[dict]:
        results = []
        for item in preview:
            result = dict(item)
            if item["status"] == "ERROR":
                self.context.store.audit("BULK_PROTECTION_FAILED", item["ticker"], result)
                results.append(result)
                continue
            state = ProtectionState(
                ticker=item["ticker"],
                stop_loss=item["stop_loss"],
                take_profit=item["take_profit"],
                trailing_stop_pct=item["trailing_stop_pct"],
                trailing_stop=item["trailing_stop"],
                highest_price=item["highest_price"],
                updated_at=self.now(),
            )
            self.context.store.save_protection(state)
            result["status"] = "WARNING" if item["status"] == "WARNING" else "SAVED"
            result["message"] = item["message"] if item["status"] == "WARNING" else "저장 완료"
            self.context.store.audit("BULK_PROTECTION_UPDATED", item["ticker"], result)
            results.append(result)
        return results

    def toggle_kill_switch(self) -> str:
        session = self._ensure_manual_session()
        value = "NORMAL" if session.get("kill_switch") == "HALTED" else "HALTED"
        self.context.store.set_session_controls(
            self.current_session_id(), kill_switch=value,
        )
        self.context.store.audit(
            "KILL_SWITCH_CHANGED", self.current_session_id(), {"kill_switch": value}
        )
        return value

    @property
    def scheduler_running(self) -> bool:
        return bool(self._scheduler_thread and self._scheduler_thread.is_alive())

    def start_scheduler(self) -> bool:
        if self.scheduler_running:
            return False
        self._scheduler_stop.clear()
        self._scheduler_error = None
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="trading-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        return True

    def _scheduler_loop(self) -> None:
        try:
            while not self._scheduler_stop.is_set():
                for result in self.service.run_due():
                    if result.status != "DUPLICATE_SKIPPED":
                        self._scheduler_last_result = result
                self._scheduler_stop.wait(1.0)
        except Exception as error:
            self._scheduler_error = f"{type(error).__name__}: {error}"

    def stop_scheduler(self) -> bool:
        if not self.scheduler_running:
            return False
        self._scheduler_stop.set()
        assert self._scheduler_thread is not None
        self._scheduler_thread.join(timeout=3)
        return True

    def run_job(self, name: str) -> JobResult:
        now = self.now()
        jobs = {
            "pre_open": self.service.run_pre_open,
            "opening_buy": self.service.run_opening_buy,
            "monitor": self.service.run_monitor,
            "post_close": self.service.run_post_close,
            "reconciliation": self.service.run_reconciliation,
        }
        if name not in jobs:
            raise TradingControlError(f"지원하지 않는 작업: {name}")
        return jobs[name](now)

    def candidates(self) -> list[dict]:
        session = self.current_session()
        return list(((session or {}).get("payload") or {}).get("candidates", []))

    def scheduled_orders(self) -> list[dict]:
        return self.context.store.list_scheduled_orders(
            statuses=("QUEUED", "EXECUTING")
        )

    def cancel_scheduled_order(self, reservation_id: str) -> dict:
        queued = {
            item["reservation_id"]: item
            for item in self.context.store.list_scheduled_orders(
                statuses=("QUEUED",)
            )
        }
        item = queued.get(reservation_id)
        if item is None:
            raise TradingControlError("취소할 수 있는 예약 주문이 없습니다.")
        if not self.context.store.cancel_scheduled_order(reservation_id):
            raise TradingControlError(
                "예약 주문이 이미 실행 중이거나 상태가 변경되었습니다."
            )
        result = {**item, "status": "CANCELLED"}
        self.context.store.audit("SCHEDULED_ORDER_CANCELLED", reservation_id, result)
        return result

    def amend_scheduled_order(self, reservation_id: str, quantity: int) -> dict:
        if quantity <= 0:
            raise TradingControlError("예약 주문 수량은 1주 이상이어야 합니다.")
        queued = {
            item["reservation_id"]: item
            for item in self.context.store.list_scheduled_orders(
                statuses=("QUEUED",)
            )
        }
        item = queued.get(reservation_id)
        if item is None:
            raise TradingControlError("정정할 수 있는 예약 주문이 없습니다.")
        if item["side"] == "SELL":
            position = self.context.broker.get_position(item["ticker"])
            held = position.quantity if position is not None else 0
            if quantity > held:
                raise TradingControlError(
                    f"예약 매도 수량이 현재 보유수량 {held:,}주를 초과합니다."
                )
        if not self.context.store.amend_scheduled_order_quantity(
            reservation_id, quantity
        ):
            raise TradingControlError(
                "예약 주문이 이미 실행 중이거나 상태가 변경되었습니다."
            )
        result = {**item, "quantity": quantity}
        result["payload"] = {**item.get("payload", {}), "quantity": quantity}
        self.context.store.audit("SCHEDULED_ORDER_AMENDED", reservation_id, result)
        return result

    def pending_orders(self) -> list[dict]:
        return self.context.store.list_reconcilable_orders()

    def api_reconciliation(self) -> dict:
        """Run broker reconciliation now, bypassing scheduler deduplication."""
        if self.service.reconciler is None:
            raise TradingControlError("API reconciliation이 설정되어 있지 않습니다.")
        try:
            report = asdict(self.service.reconciler.reconcile())
        except Exception as error:
            self.context.store.audit(
                "MANUAL_RECONCILIATION_FAILED",
                self.current_session_id(),
                {"error": f"{type(error).__name__}: {error}"},
            )
            raise TradingControlError(f"API reconciliation 실패: {error}") from error
        self.context.store.audit(
            "MANUAL_RECONCILIATION", self.current_session_id(), report
        )
        return report

    def cancel_pending_order(self, idempotency_key: str) -> dict:
        """Request a full cancellation while preserving the original order identity."""
        self._validate_order_window()
        pending = {
            item["idempotency_key"]: item
            for item in self.context.store.list_reconcilable_orders()
        }
        intent = pending.get(idempotency_key)
        if intent is None:
            raise TradingControlError("취소할 수 있는 미체결 주문을 찾지 못했습니다.")

        if self.context.config.dry_run:
            result = OrderResult(
                status="DRY_RUN",
                ticker=intent["ticker"],
                side="CANCEL",
                quantity=int(intent.get("payload", {}).get("remaining_quantity") or 0),
                reason="Dry run: 취소 주문을 KIS에 전송하지 않았습니다.",
            )
        else:
            try:
                result = self.context.broker.cancel_order(
                    intent["broker_order_id"],
                    intent["session_id"][:10],
                    intent["ticker"],
                )
            except NotImplementedError as error:
                raise TradingControlError(
                    "현재 Broker는 주문 취소를 지원하지 않습니다."
                ) from error

        payload = {
            "cancel_request": asdict(result),
            "cancel_requested_at": self.now().isoformat(),
            "original_status": intent["status"],
        }
        # 취소 접수는 최종 취소가 아니다. 원 주문 ID와 상태를 유지해야
        # 후속 KIS 체결조회에서 CANCELLED 여부를 확정할 수 있다.
        self.context.store.update_order_intent(
            idempotency_key,
            intent["status"],
            broker_order_id=intent["broker_order_id"],
            payload=payload,
        )
        event = (
            "ORDER_CANCEL_REQUESTED"
            if result.status == "CANCEL_SUBMITTED"
            else "ORDER_CANCEL_REJECTED"
        )
        self.context.store.audit(event, idempotency_key, payload)

        if result.status == "CANCEL_SUBMITTED" and self.service.reconciler is not None:
            self.service.reconciler.reconcile()
        current = next(
            (
                item for item in self.context.store.list_order_intents(limit=100)
                if item["idempotency_key"] == idempotency_key
            ),
            intent,
        )
        return {"request": asdict(result), "order": current}

    def order_history(self) -> list[dict]:
        return self.context.store.list_order_intents(limit=50)

    def audit_history(self) -> list[dict]:
        return self.context.store.list_audit_events(limit=50)

    def test_notification(self) -> None:
        if self.service.reconciler is None or not self.service.reconciler.notifier.enabled:
            raise TradingControlError("Slack 알림이 활성화되어 있지 않습니다.")
        self.service.reconciler.notifier.send(
            ":white_check_mark: *KRX 자동매매 Slack 연결 테스트 성공*"
        )

    def top_recommendations(
        self,
        *,
        universe_scope: str = "BOTH",
        refresh: bool = False,
        progress=None,
    ) -> list[dict]:
        scope = universe_scope.strip().upper()
        selected_limit = self.context.config.recommendation_final_limit
        recommendations = self.recommendation_service.top_recommendations(
            self.now().date(), limit=selected_limit, universe_scope=universe_scope,
            refresh=refresh, progress=progress
        )
        if recommendations:
            trade_date = self.now().date().isoformat()
            report_path = render_top10_pick_report(
                recommendations,
                trade_date=trade_date,
                universe_scope=scope,
                directory=self.context.config.rebalance_report_dir,
            )
            saved = {
                "trade_date": trade_date,
                "universe_scope": scope,
                "recommendations": recommendations,
                "report_path": str(report_path.resolve()),
            }
            if self.context.config.rebalance_report_base_url:
                saved["report_url"] = self.context.config.rebalance_report_base_url
            # Persist first so menu 13 can reuse a completed analysis even if
            # the optional Slack delivery fails afterward.
            self.context.store.set_control("top_recommendations_latest", saved)
            report_url = self._notify_top10_pick(saved)
            if report_url:
                saved["report_url"] = report_url
                self.context.store.set_control("top_recommendations_latest", saved)
        return recommendations

    def _notify_top10_pick(self, result: dict) -> str:
        """Send the OneDrive PDF location to Slack, with upload as fallback."""
        notifier = (
            self.service.reconciler.notifier if self.service.reconciler else None
        )
        if not notifier or not notifier.enabled:
            return ""
        report_path = Path(result["report_path"])
        report_url = str(result.get("report_url") or "")
        upload = getattr(notifier, "upload_file", None)
        if callable(upload) and not report_url:
            try:
                report_url = upload(
                    report_path,
                    title=f"오늘의 Top10 pick {result['trade_date']} {result['universe_scope']}",
                    initial_comment=(
                        f"오늘의 Top10 pick 분석 결과 · {result['universe_scope']}"
                    ),
                ) or ""
            except Exception as error:
                self.context.store.audit(
                    "TOP10_REPORT_UPLOAD_FAILED",
                    result["trade_date"],
                    {"error": f"{type(error).__name__}: {error}"},
                )
        if not report_url:
            report_url = self.context.config.rebalance_report_base_url
        link = (
            f"\n• OneDrive PDF `{report_path.name}`: {report_url}" if report_url else
            f"\n• PDF 저장 위치: `{report_path}`"
        )
        notifier.send(
            ":bar_chart: *오늘의 Top10 pick 분석 완료*\n"
            f"• 기준일: {result['trade_date']}\n"
            f"• 유니버스: {result['universe_scope']}\n"
            f"• 추천종목: {len(result['recommendations'])}개"
            + link
        )
        return report_url

    def latest_top_recommendations(self) -> tuple[str, list[dict]]:
        """Return the exact Top10 result last completed from menu 12 today."""
        saved = self.context.store.get_control("top_recommendations_latest")
        today = self.now().date().isoformat()
        if not saved or saved.get("trade_date") != today:
            raise TradingControlError(
                "오늘 10번 메뉴에서 완료한 Top10 결과가 없습니다. "
                "먼저 분석 유니버스를 선택해 10번을 실행하세요."
            )
        recommendations = saved.get("recommendations") or []
        if not recommendations:
            raise TradingControlError(
                "10번 메뉴의 저장된 Top10 결과가 비어 있습니다. 다시 분석하세요."
            )
        scope = str(saved.get("universe_scope") or "BOTH").upper()
        return scope, recommendations

    def rebalance_proposal(self, *, progress=None) -> dict:
        if not self.context.config.rebalance_enabled:
            raise TradingControlError("LLM 리밸런싱이 비활성화되어 있습니다.")
        saved = self.context.store.get_control("rebalance_latest")
        if (
            saved
            and str(saved.get("created_at") or "")[:10]
            == self.now().date().isoformat()
        ):
            return saved
        if not self.context.config.rebalance_enabled:
            raise TradingControlError(
                "LLM 리밸런싱이 비활성화되어 있습니다. "
                "TRADING_REBALANCE_ENABLED=true 설정이 필요합니다."
            )
        balance, positions = self.account_snapshot()
        position_rows = []
        total_market_value = 0.0
        for ticker, position in positions.items():
            current = self.quote(ticker)
            market_value = current * position.quantity
            total_market_value += market_value
            position_rows.append({
                "ticker": ticker,
                "name": stock_name(ticker),
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "current_price": current,
                "market_value": market_value,
                "return_pct": (
                    (current - position.avg_price) / position.avg_price * 100
                    if position.avg_price > 0 else 0
                ),
                "sector": position.sector or "UNKNOWN",
            })
        total_equity = float(balance.get("total_equity") or 0)
        if total_equity <= 0:
            total_equity = float(balance.get("cash") or 0) + total_market_value
        for item in position_rows:
            item["weight_pct"] = (
                item["market_value"] / total_equity * 100 if total_equity else 0
            )
        recommendation_scope, top10 = self.latest_top_recommendations()
        normalized_top10 = []
        for item in top10:
            row = dict(item)
            row["current_price"] = self.quote(row["ticker"])
            normalized_top10.append(row)
        market_news = self.market_news_service.collect()
        if not market_news.get("headlines"):
            raise TradingControlError(
                "오늘의 시장뉴스를 수집하지 못했습니다. 뉴스 API 설정을 확인하세요."
            )
        security_collector = getattr(
            self.market_news_service, "collect_securities", None
        )
        security_news = (
            security_collector(position_rows + normalized_top10)
            if callable(security_collector) else {}
        )
        snapshot = {
            "as_of": self.now().isoformat(),
            "recommendation_universe_scope": recommendation_scope,
            "portfolio": {
                "cash": float(balance.get("cash") or 0),
                "total_equity": total_equity,
                "market_value": total_market_value,
                "portfolio_return_pct": (
                    float(balance.get("unrealized_pnl") or 0) /
                    max(total_equity - float(balance.get("unrealized_pnl") or 0), 1)
                    * 100
                ),
            },
            "positions": position_rows,
            "top10": normalized_top10,
            "market_news": market_news,
            "security_news": security_news,
        }
        advisor = self.rebalance_advisor or LLMRebalanceAdvisor(
            model=self.context.config.rebalance_llm_model
        )
        proposal = advisor.propose(snapshot)
        validation = RebalanceValidator(self.context.config).validate(
            snapshot, proposal
        )
        package = {
            "created_at": self.now().isoformat(),
            "snapshot": snapshot,
            "proposal": proposal.model_dump(),
            "validation": validation,
            "requires_individual_review": True,
        }
        package["proposal_id"] = proposal_id(package)
        report_path = render_rebalance_report(
            package, self.context.config.rebalance_report_dir
        )
        package["report_path"] = str(report_path.resolve())
        if self.context.config.rebalance_report_base_url:
            package["report_url"] = self.context.config.rebalance_report_base_url
        self.context.store.set_control("rebalance_latest", package)
        self.context.store.audit(
            "REBALANCE_PROPOSED", package["proposal_id"], package
        )
        report_url = self._notify_rebalance_proposal(package)
        if report_url:
            package["report_url"] = report_url
            self.context.store.set_control("rebalance_latest", package)
        return package

    def revise_rebalance_proposal(
        self, expected_proposal_id: str, user_feedback: str
    ) -> dict:
        package = self.context.store.get_control("rebalance_latest")
        if not package or package.get("proposal_id") != expected_proposal_id:
            raise TradingControlError("저장된 리밸런싱 제안서 ID가 일치하지 않습니다.")
        feedback = user_feedback.strip()
        if not feedback:
            raise TradingControlError("LLM에 전달할 수정 의견을 입력하세요.")
        advisor = self.rebalance_advisor or LLMRebalanceAdvisor(
            model=self.context.config.rebalance_llm_model
        )
        revise = getattr(advisor, "revise", None)
        if not callable(revise):
            raise TradingControlError("현재 Advisor는 제안 수정 기능을 지원하지 않습니다.")
        revised = revise(package["snapshot"], package["proposal"], feedback)
        revised = (
            revised if isinstance(revised, RebalanceProposal)
            else RebalanceProposal.model_validate(revised)
        )
        validation = RebalanceValidator(self.context.config).validate(
            package["snapshot"], revised
        )
        history = list(package.get("revision_history") or [])
        history.append({
            "proposal_id": package["proposal_id"],
            "proposal": package["proposal"],
            "validation": package["validation"],
            "feedback": feedback,
            "revised_at": self.now().isoformat(),
        })
        updated = {
            **package,
            "created_at": self.now().isoformat(),
            "proposal": revised.model_dump(),
            "validation": validation,
            "requires_individual_review": True,
            "revision": len(history),
            "revision_history": history,
        }
        updated.pop("individual_review", None)
        updated["proposal_id"] = proposal_id({
            "created_at": updated["created_at"],
            "snapshot": updated["snapshot"],
            "proposal": updated["proposal"],
            "revision": updated["revision"],
        })
        report_path = render_rebalance_report(
            updated, self.context.config.rebalance_report_dir
        )
        updated["report_path"] = str(report_path.resolve())
        self.context.store.set_control("rebalance_latest", updated)
        self.context.store.audit(
            "REBALANCE_REVISED", updated["proposal_id"],
            {"previous_proposal_id": expected_proposal_id, "feedback": feedback},
        )
        report_url = self._notify_rebalance_proposal(updated)
        if report_url:
            updated["report_url"] = report_url
            self.context.store.set_control("rebalance_latest", updated)
        return updated

    def review_rebalance_orders(
        self,
        expected_proposal_id: str,
        reviewed_orders: list[dict],
    ) -> dict:
        """Persist per-security approvals/edits after deterministic revalidation."""
        package = self.context.store.get_control("rebalance_latest")
        if not package or package.get("proposal_id") != expected_proposal_id:
            raise TradingControlError("리밸런싱 제안서 ID가 일치하지 않습니다.")
        allowed = {
            item["ticker"] for item in package["validation"].get("orders", [])
        }
        unexpected = {
            str(item.get("ticker") or "").upper() for item in reviewed_orders
        } - allowed
        if unexpected:
            raise TradingControlError(
                f"원 제안에 없는 종목은 추가할 수 없습니다: {sorted(unexpected)}"
            )
        proposal = RebalanceProposal.model_validate(package["proposal"])
        validation = RebalanceValidator(self.context.config).validate_reviewed_orders(
            package["snapshot"], proposal, reviewed_orders
        )
        package["validation"] = validation
        package["individual_review"] = {
            "reviewed_at": self.now().isoformat(),
            "original_order_count": len(allowed),
            "approved_order_count": len(validation["orders"]),
            "orders": reviewed_orders,
        }
        self.context.store.set_control("rebalance_latest", package)
        self.context.store.audit(
            "REBALANCE_ORDERS_REVIEWED", expected_proposal_id,
            package["individual_review"],
        )
        return package

    def execute_rebalance(
        self,
        expected_proposal_id: str,
        *,
        override_risk: bool = False,
    ) -> dict:
        if not self.context.config.rebalance_enabled:
            raise TradingControlError("LLM 리밸런싱이 비활성화되어 있습니다.")
        package = self.context.store.get_control("rebalance_latest")
        if not package or package.get("proposal_id") != expected_proposal_id:
            raise TradingControlError("리밸런싱 제안서 ID가 일치하지 않습니다.")
        created_at = datetime.fromisoformat(package["created_at"])
        age_minutes = (self.now() - created_at).total_seconds() / 60
        if (
            age_minutes > self.context.config.rebalance_proposal_ttl_minutes
            and created_at.date() != self.now().date()
        ):
            raise TradingControlError("리밸런싱 제안서가 만료되었습니다. 다시 분석하세요.")
        validation = package["validation"]
        if package.get("requires_individual_review") and not validation.get(
            "individually_reviewed", False
        ):
            raise TradingControlError(
                "종목별 주문 승인 또는 수정 절차가 완료되지 않았습니다."
            )
        session = self._ensure_manual_session()
        has_buy_orders = any(
            item.get("side") == "BUY" for item in validation.get("orders", [])
        )
        if session.get("kill_switch") == "HALTED":
            raise TradingControlError(
                "Kill Switch가 활성화되어 있습니다. 16번 메뉴에서 NORMAL로 전환한 "
                "뒤 리밸런싱을 다시 실행하세요. 주문은 시작되지 않았습니다."
            )
        if not validation.get("approved"):
            if not override_risk:
                raise TradingControlError(
                    "Risk Validator 거부 제안은 Override 승인이 필요합니다."
                )
            if not validation.get("override_allowed"):
                raise TradingControlError(
                    "주문 무결성 관련 거부 사유가 있어 Override할 수 없습니다."
                )
            override_payload = {
                "proposal_id": expected_proposal_id,
                "errors": validation.get("errors", []),
                "confirmed_at": self.now().isoformat(),
            }
            self.context.store.audit(
                "REBALANCE_RISK_OVERRIDE", expected_proposal_id, override_payload
            )
            notifier = (
                self.service.reconciler.notifier if self.service.reconciler else None
            )
            if notifier and notifier.enabled:
                notifier.send(
                    ":warning: *리밸런싱 Risk Override 승인*\n"
                    f"• 제안서: `{expected_proposal_id}`\n"
                    + "\n".join(
                        f"• Override 사유: {error}"
                        for error in validation.get("errors", [])
                    )
                )
        execution_key = f"rebalance_execution:{expected_proposal_id}"
        previous_execution = self.context.store.get_control(execution_key)
        retryable = {"FAILED", "AWAITING_SELL_FILLS"}
        if (
            previous_execution is not None
            and previous_execution.get("status") not in retryable
        ):
            raise TradingControlError("이미 실행했거나 실행 중인 리밸런싱 제안서입니다.")
        phase = self.context.calendar.phase(self.now())
        if phase in {"CLOSED_DAY", "PRE_OPEN", "REGULAR", "POST_CLOSE"}:
            result = self._schedule_rebalance(package)
            self.context.store.set_control(
                execution_key, {**result, "reserved_at": self.now().isoformat()}
            )
            self.context.store.audit(
                "REBALANCE_RESERVED", expected_proposal_id,
                {"risk_override": override_risk, **result},
            )
            return result
        self.context.store.set_control(
            execution_key, {"status": "STARTED", "started_at": self.now().isoformat()}
        )
        try:
            result = RebalanceExecutor(self).execute(package)
        except Exception as error:
            failed = {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "risk_override": override_risk,
                "finished_at": self.now().isoformat(),
            }
            self.context.store.set_control(execution_key, failed)
            self.context.store.audit(
                "REBALANCE_FAILED", expected_proposal_id, failed
            )
            raise
        self.context.store.set_control(
            execution_key, {**result, "finished_at": self.now().isoformat()}
        )
        payload = {
            "proposal_id": expected_proposal_id,
            "risk_override": override_risk,
            **result,
        }
        self.context.store.audit(
            "REBALANCE_EXECUTED", expected_proposal_id, payload
        )
        notifier = (
            self.service.reconciler.notifier if self.service.reconciler else None
        )
        if notifier and notifier.enabled:
            lines = [
                f":white_check_mark: *리밸런싱 {result['status']}*",
                f"• 제안서: `{expected_proposal_id}`",
                f"• Risk Override: {'YES' if override_risk else 'NO'}",
            ]
            lines.extend(
                f"• {item['side']} {item['ticker']} {item['quantity']:,}주: {item['status']}"
                for item in result["orders"]
            )
            notifier.send("\n".join(lines))
        return result

    def _schedule_rebalance(self, package: dict) -> dict:
        now = self.now().astimezone(self.context.config.timezone)
        phase = self.context.calendar.phase(now)
        execute_date = self.context.calendar.next_trading_day(
            now.date(), include_today=phase in {"PRE_OPEN", "REGULAR"}
        )
        queued = []
        for item in package["validation"].get("orders", []):
            reservation_id = (
                f"{package['proposal_id']}:{item['side']}:{item['ticker']}"
            )
            payload = {
                **item,
                "proposal_id": package["proposal_id"],
                "approved_at": now.isoformat(),
                "order_type": item.get("order_type") or "PRIORITY_LIMIT",
            }
            created = self.context.store.enqueue_scheduled_order(
                reservation_id,
                package["proposal_id"],
                execute_date.isoformat(),
                item["ticker"],
                item["side"],
                int(item["quantity"]),
                payload,
            )
            if created:
                queued.append({
                    **item, "reservation_id": reservation_id,
                    "execute_on": execute_date.isoformat(), "status": "QUEUED",
                })
        return {
            "status": "ORDERS_RESERVED",
            "execute_on": execute_date.isoformat(),
            "orders": queued,
        }

    def _notify_rebalance_proposal(self, package: dict) -> str:
        notifier = (
            self.service.reconciler.notifier if self.service.reconciler else None
        )
        if not notifier or not notifier.enabled:
            return ""
        proposal = package["proposal"]
        validation = package["validation"]
        validator_status = (
            "통과" if validation["approved"] else (
                "거부(Override 가능)"
                if validation.get("override_allowed") else "거부(Override 불가)"
            )
        )
        report_url = str(package.get("report_url") or "")
        upload = getattr(notifier, "upload_file", None)
        if callable(upload) and not report_url:
            try:
                uploaded_url = upload(
                    Path(package["report_path"]),
                    title=f"리밸런싱 리포트 {package['proposal_id']}",
                    initial_comment=f"LLM 리밸런싱 상세 리포트 `{package['proposal_id']}`",
                )
                report_url = uploaded_url or report_url
            except Exception as error:
                self.context.store.audit(
                    "REBALANCE_REPORT_UPLOAD_FAILED", package["proposal_id"],
                    {"error": f"{type(error).__name__}: {error}"},
                )
        link_line = (
            f"\n• OneDrive PDF 제안서 `{Path(package['report_path']).name}`: {report_url}"
            if report_url else
            f"\n• PDF 제안서 저장 위치: `{package.get('report_path', '-')}`"
        )
        notifier.send(
            ":robot_face: *LLM 리밸런싱 제안 생성*\n"
            f"• 제안서: `{package['proposal_id']}`\n"
            f"• 시장 판단: {proposal['market_view']}\n"
            f"• 권장 현금: {proposal['recommended_cash_pct']:.1f}%\n"
            f"• 주문 후보: {len(validation['orders'])}건\n"
            f"• Risk Validator: {validator_status}"
            + link_line
        )
        return report_url
