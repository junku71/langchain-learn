from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from broker.base import Broker
from paper.trade_logger import TradeLogger
from portfolio_manager import PortfolioLimits, PortfolioManager, can_add_position
from risk.risk_config import RiskConfig
from risk.risk_engine import calculate_position_risk
from trading.calendar import KrxCalendar
from trading.config import LiveTradingConfig
from trading.models import ProtectionState
from trading.trade_logging import log_trade_safely
from trading.protection import (
    evaluate_minute_bar,
    latest_donchian_low,
    protection_from_position,
)
from trading.providers import CandidateAnalysis, CandidateProvider, QuoteProvider
from trading.state_store import TradingStateStore


class TradingGraphState(TypedDict, total=False):
    now: datetime
    trade_date: str
    session_id: str
    status: str
    reason: str
    buy_enabled: bool
    kill_switch: str
    candidates: list[dict]
    candidate_index: int
    current_candidate: dict | None
    bar: Any
    analysis_result: dict | None
    risk_result: dict | None
    portfolio_result: dict | None
    accepted_orders: list[dict]
    rejected_candidates: list[dict]
    monitor_results: list[dict]
    report: dict


@dataclass
class TradingGraphContext:
    config: LiveTradingConfig
    calendar: KrxCalendar
    store: TradingStateStore
    broker: Broker
    quote_provider: QuoteProvider
    candidate_provider: CandidateProvider
    analysis: CandidateAnalysis
    trade_logger: TradeLogger

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return PortfolioManager(self.broker)


def _session_id(trade_date: date, strategy_version: str) -> str:
    return f"{trade_date.isoformat()}:{strategy_version}"


def _current_prices(context: TradingGraphContext, now: datetime) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker, position in context.broker.get_positions().items():
        try:
            prices[ticker] = context.quote_provider.minute_bar(ticker, now).close
        except (RuntimeError, ValueError):
            prices[ticker] = position.avg_price
    return prices


def build_pre_open_graph(context: TradingGraphContext):
    builder = StateGraph(TradingGraphState)

    def session_guard(state: TradingGraphState) -> dict:
        now = state["now"].astimezone(context.config.timezone)
        trade_date = now.date()
        session_id = _session_id(trade_date, context.config.strategy_version)
        if not context.calendar.is_trading_day(trade_date):
            return {
                "trade_date": trade_date.isoformat(), "session_id": session_id,
                "status": "SKIPPED", "reason": "NOT_TRADING_DAY",
                "buy_enabled": True,
            }
        return {
            "trade_date": trade_date.isoformat(), "session_id": session_id,
            "status": "CALENDAR_OK", "buy_enabled": True,
        }

    def route_calendar(state: TradingGraphState) -> str:
        return "prepare" if state["status"] == "CALENDAR_OK" else "skip"

    def prepare(state: TradingGraphState) -> dict:
        try:
            trade_date = date.fromisoformat(state["trade_date"])
            balance = context.broker.get_balance()
            candidates = context.candidate_provider.candidates(
                trade_date, context.config.max_candidates_per_market
            )
            payload = {
                "candidates": candidates,
                "starting_balance": balance,
                "prepared_at": state["now"].isoformat(),
            }
            context.store.upsert_session(
                state["session_id"], state["trade_date"],
                context.config.strategy_version, "SESSION_READY",
                buy_enabled=True, payload=payload,
            )
            context.store.audit("SESSION_READY", state["session_id"], payload)
            return {
                "status": "SESSION_READY", "buy_enabled": True,
                "candidates": candidates,
            }
        except Exception as error:
            payload = {"error": f"{type(error).__name__}: {error}"}
            context.store.upsert_session(
                state["session_id"], state["trade_date"],
                context.config.strategy_version, "BUY_DISABLED",
                buy_enabled=True, kill_switch="BUY_DISABLED", payload=payload,
            )
            context.store.audit("PRE_OPEN_FAILED", state["session_id"], payload)
            return {
                "status": "BUY_DISABLED", "buy_enabled": True,
                "reason": payload["error"],
            }

    def record_skip(state: TradingGraphState) -> dict:
        context.store.audit("SESSION_SKIPPED", state["session_id"], {
            "reason": state.get("reason", "UNKNOWN")
        })
        return {}

    builder.add_node("session_guard", session_guard)
    builder.add_node("prepare", prepare)
    builder.add_node("record_skip", record_skip)
    builder.add_edge(START, "session_guard")
    builder.add_conditional_edges(
        "session_guard", route_calendar,
        {"prepare": "prepare", "skip": "record_skip"},
    )
    builder.add_edge("prepare", END)
    builder.add_edge("record_skip", END)
    return builder.compile()


def build_opening_buy_graph(context: TradingGraphContext):
    builder = StateGraph(TradingGraphState)

    def load_session(state: TradingGraphState) -> dict:
        now = state["now"].astimezone(context.config.timezone)
        session_id = _session_id(now.date(), context.config.strategy_version)
        session = context.store.get_session(session_id)
        session_open = context.calendar.phase(now) == "REGULAR"
        enabled = bool(
            session and session["kill_switch"] == "NORMAL" and session_open
        )
        return {
            "session_id": session_id, "trade_date": now.date().isoformat(),
            "buy_enabled": enabled,
            "status": "READY" if enabled else "SKIPPED",
            "reason": "" if enabled else "SESSION_NOT_READY",
            "candidates": (session or {}).get("payload", {}).get("candidates", []),
            "candidate_index": 0, "accepted_orders": [],
            "rejected_candidates": [],
        }

    def route_start(state: TradingGraphState) -> str:
        return "next" if state["buy_enabled"] else "finish"

    def select_candidate(state: TradingGraphState) -> dict:
        index = state["candidate_index"]
        candidates = state["candidates"]
        if (
            index >= len(candidates)
            or len(state["accepted_orders"]) >= context.config.max_new_positions_per_day
            or context.store.count_orders(state["session_id"]) >= context.config.max_daily_orders
        ):
            return {"current_candidate": None}
        return {"current_candidate": candidates[index]}

    def route_candidate(state: TradingGraphState) -> str:
        return "evaluate" if state.get("current_candidate") else "finish"

    def evaluate_candidate(state: TradingGraphState) -> dict:
        candidate = state["current_candidate"]
        ticker = candidate["ticker"]
        if context.broker.get_position(ticker) is not None:
            return {"analysis_result": {"approved": False, "reason": "ALREADY_HELD"}}
        try:
            bar = context.quote_provider.minute_bar(ticker, state["now"])
            if not bar.valid():
                raise ValueError("invalid quote")
            result = context.analysis.evaluate(candidate, bar)
            analysis_approved = bool(result.get("approved", False))
            ml_filter_enabled = context.store.get_bool_control(
                "ml_filter_enabled", context.config.ml_filter_enabled
            )
            approved = analysis_approved
            if analysis_approved and ml_filter_enabled:
                approved = (
                    float(result["classification_probability"])
                    >= context.config.ml_probability_threshold
                    and int(result["ml_rank"])
                    <= context.config.max_candidates_per_market
                )
            result["approved"] = approved
            result["ml_filter_enabled"] = ml_filter_enabled
            if analysis_approved and not approved:
                result["reason"] = "ML_GUARD_REJECTED"
            elif analysis_approved and not ml_filter_enabled:
                result["reason"] = "ML_FILTER_BYPASSED"
            return {"bar": bar, "analysis_result": result}
        except Exception as error:
            return {"analysis_result": {
                "approved": False,
                "reason": f"QUOTE_OR_ANALYSIS_ERROR: {type(error).__name__}: {error}",
            }}

    def route_analysis(state: TradingGraphState) -> str:
        return "risk" if state["analysis_result"]["approved"] else "reject"

    def calculate_risk(state: TradingGraphState) -> dict:
        candidate = state["current_candidate"]
        price = float(state["bar"].close)
        atr_pct = pd.to_numeric(candidate.get("atr_pct"), errors="coerce")
        if pd.isna(atr_pct) or atr_pct <= 0:
            return {"risk_result": {"approved": False, "reason": "ATR_MISSING"}}
        balance = context.broker.get_balance()
        account_size = float(
            balance.get("total_equity")
            or balance.get("cash")
            or balance.get("initial_cash", 0)
        )
        risk = calculate_position_risk(
            price=price, atr=price * float(atr_pct), account_size=account_size,
            config=RiskConfig(),
        )
        return {"risk_result": risk}

    def route_risk(state: TradingGraphState) -> str:
        return "portfolio" if state["risk_result"].get("approved") else "reject"

    def portfolio_guard(state: TradingGraphState) -> dict:
        prices = _current_prices(context, state["now"])
        report = context.portfolio_manager.evaluate(prices)
        guard = can_add_position(
            portfolio_report=report,
            ticker=state["current_candidate"]["ticker"],
            sector=state["current_candidate"].get("sector", "UNKNOWN"),
            new_position_value=state["risk_result"]["position_value"],
            limits=PortfolioLimits(),
        )
        guard["portfolio_report"] = report
        return {"portfolio_result": guard}

    def route_portfolio(state: TradingGraphState) -> str:
        return "order" if state["portfolio_result"].get("approved") else "reject"

    def submit_order(state: TradingGraphState) -> dict:
        candidate = state["current_candidate"]
        risk = state["risk_result"]
        key = f"{state['session_id']}:{candidate['ticker']}:BUY"
        intent = {
            "price": risk["price"], "quantity": risk["position_size"],
            "stop_loss": risk["stop_loss"], "take_profit": risk["take_profit"],
            "trailing_stop_pct": context.config.trailing_stop_pct,
            "candidate": candidate,
        }
        if not context.store.create_order_intent(
            key, state["session_id"], candidate["ticker"], "BUY", intent
        ):
            return {"analysis_result": {"approved": False, "reason": "DUPLICATE_ORDER"}}

        if context.config.dry_run:
            result = {
                "status": "DRY_RUN", "ticker": candidate["ticker"],
                "side": "BUY", "price": risk["price"],
                "quantity": risk["position_size"], "reason": "DRY_RUN",
            }
        else:
            order = context.broker.buy(
                ticker=candidate["ticker"], price=risk["price"],
                quantity=risk["position_size"],
                sector=candidate.get("sector", "UNKNOWN"),
                stop_loss=risk["stop_loss"], take_profit=risk["take_profit"],
                trailing_stop_pct=context.config.trailing_stop_pct,
                reason="OPENING_GRAPH_BUY",
            )
            result = asdict(order)
            context.store.update_order_intent(
                key, result["status"], broker_order_id=result.get("order_id", ""),
                payload=result,
            )
            log_trade_safely(context, order, entity_key=key)
            position = context.broker.get_position(candidate["ticker"])
            if position is not None:
                context.store.save_protection(
                    protection_from_position(position, state["now"])
                )
        if context.config.dry_run:
            context.store.update_order_intent(
                key, result["status"], broker_order_id=result.get("order_id", ""),
                payload=result,
            )
        context.store.audit("BUY_ORDER", key, result)
        accepted = [*state["accepted_orders"], result]
        return {"accepted_orders": accepted, "analysis_result": {"approved": True}}

    def reject_candidate(state: TradingGraphState) -> dict:
        candidate = state["current_candidate"]
        reason = (
            (state.get("analysis_result") or {}).get("reason")
            or (state.get("risk_result") or {}).get("reason")
            or (state.get("portfolio_result") or {}).get("reason")
            or "REJECTED"
        )
        rejected = [*state["rejected_candidates"], {
            "ticker": candidate["ticker"], "reason": reason,
        }]
        context.store.audit("CANDIDATE_REJECTED", candidate["ticker"], {
            "session_id": state["session_id"], "reason": reason,
        })
        return {"rejected_candidates": rejected}

    def advance(state: TradingGraphState) -> dict:
        return {
            "candidate_index": state["candidate_index"] + 1,
            "current_candidate": None, "bar": None,
            "analysis_result": None, "risk_result": None,
            "portfolio_result": None,
        }

    def finish(state: TradingGraphState) -> dict:
        report = {
            "status": state.get("status", "COMPLETED"),
            "accepted": state.get("accepted_orders", []),
            "rejected": state.get("rejected_candidates", []),
        }
        context.store.audit("OPENING_BUY_COMPLETE", state["session_id"], report)
        return {"status": "COMPLETED", "report": report}

    for name, node in (
        ("load_session", load_session), ("select_candidate", select_candidate),
        ("evaluate_candidate", evaluate_candidate), ("calculate_risk", calculate_risk),
        ("portfolio_guard", portfolio_guard), ("submit_order", submit_order),
        ("reject_candidate", reject_candidate), ("advance", advance),
        ("finish", finish),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "load_session")
    builder.add_conditional_edges("load_session", route_start, {
        "next": "select_candidate", "finish": "finish",
    })
    builder.add_conditional_edges("select_candidate", route_candidate, {
        "evaluate": "evaluate_candidate", "finish": "finish",
    })
    builder.add_conditional_edges("evaluate_candidate", route_analysis, {
        "risk": "calculate_risk", "reject": "reject_candidate",
    })
    builder.add_conditional_edges("calculate_risk", route_risk, {
        "portfolio": "portfolio_guard", "reject": "reject_candidate",
    })
    builder.add_conditional_edges("portfolio_guard", route_portfolio, {
        "order": "submit_order", "reject": "reject_candidate",
    })
    builder.add_edge("submit_order", "advance")
    builder.add_edge("reject_candidate", "advance")
    builder.add_edge("advance", "select_candidate")
    builder.add_edge("finish", END)
    return builder.compile()


def build_position_monitor_graph(context: TradingGraphContext):
    builder = StateGraph(TradingGraphState)

    def session_guard(state: TradingGraphState) -> dict:
        now = state["now"].astimezone(context.config.timezone)
        session_id = _session_id(now.date(), context.config.strategy_version)
        session = context.store.get_session(session_id)
        allowed = (
            context.calendar.phase(now) == "REGULAR"
            and (not session or session["kill_switch"] != "HALTED")
        )
        return {
            "session_id": session_id, "trade_date": now.date().isoformat(),
            "status": "READY" if allowed else "SKIPPED",
            "monitor_results": [],
        }

    def route_session(state: TradingGraphState) -> str:
        return "monitor" if state["status"] == "READY" else "finish"

    def monitor_positions(state: TradingGraphState) -> dict:
        results: list[dict] = []
        for ticker, position in list(context.broker.get_positions().items()):
            try:
                bar = context.quote_provider.minute_bar(ticker, state["now"])
                age = abs((state["now"] - bar.timestamp).total_seconds())
                if not bar.valid() or age > context.config.quote_stale_seconds:
                    raise ValueError(f"stale or invalid quote age={age:.0f}s")
                protection = (
                    context.store.get_protection(ticker)
                    or protection_from_position(position, state["now"])
                )
                if protection.strategy == "DONCHIAN_TREND":
                    period = protection.donchian_period or 20
                    refresh_key = (
                        f"donchian_refresh:{state['trade_date']}:{ticker}:{period}"
                    )
                    if context.store.get_control(refresh_key) is None:
                        channel_low = latest_donchian_low(ticker, period)
                        protection.stop_loss = channel_low
                        protection.trailing_stop = channel_low
                        protection.updated_at = state["now"]
                        context.store.save_protection(protection)
                        context.store.set_control(
                            refresh_key,
                            {"channel_low": channel_low, "period": period},
                        )
                action, trigger_price, protection = evaluate_minute_bar(
                    protection, bar
                )
                if action == "HOLD":
                    context.store.save_protection(protection)
                    results.append({
                        "ticker": ticker, "action": "HOLD", "close": bar.close,
                        "quote_source": bar.source,
                        "effective_stop": protection.effective_stop(),
                        "take_profit": protection.take_profit,
                    })
                    continue

                key = f"{state['session_id']}:{ticker}:SELL"
                intent = {
                    "reason": action, "trigger_price": trigger_price,
                    "quantity": position.quantity, "bar": asdict(bar),
                }
                if not context.store.create_order_intent(
                    key, state["session_id"], ticker, "SELL", intent
                ):
                    results.append({
                        "ticker": ticker, "action": "SELL_PENDING",
                        "reason": "DUPLICATE_EXIT_INTENT",
                    })
                    continue
                if context.config.dry_run:
                    result = {
                        "status": "DRY_RUN", "ticker": ticker, "side": "SELL",
                        "price": trigger_price, "quantity": position.quantity,
                        "reason": action,
                    }
                else:
                    order = context.broker.sell(
                        ticker=ticker, price=float(trigger_price),
                        quantity=position.quantity, reason=action,
                    )
                    result = asdict(order)
                    context.store.update_order_intent(
                        key, result["status"],
                        broker_order_id=result.get("order_id", ""), payload=result,
                    )
                    log_trade_safely(context, order, entity_key=key)
                if context.config.dry_run:
                    context.store.update_order_intent(
                        key, result["status"],
                        broker_order_id=result.get("order_id", ""), payload=result,
                    )
                if context.broker.get_position(ticker) is None:
                    context.store.delete_protection(ticker)
                context.store.audit("SELL_ORDER", key, result)
                results.append({"ticker": ticker, "action": "SELL", **result})
            except Exception as error:
                result = {
                    "ticker": ticker, "action": "ERROR",
                    "reason": f"{type(error).__name__}: {error}",
                }
                context.store.audit("MONITOR_ERROR", ticker, result)
                results.append(result)
        return {"monitor_results": results}

    def finish(state: TradingGraphState) -> dict:
        report = {
            "status": state["status"],
            "results": state.get("monitor_results", []),
        }
        context.store.audit("MONITOR_COMPLETE", state["session_id"], report)
        return {"report": report}

    builder.add_node("session_guard", session_guard)
    builder.add_node("monitor_positions", monitor_positions)
    builder.add_node("finish", finish)
    builder.add_edge(START, "session_guard")
    builder.add_conditional_edges("session_guard", route_session, {
        "monitor": "monitor_positions", "finish": "finish",
    })
    builder.add_edge("monitor_positions", "finish")
    builder.add_edge("finish", END)
    return builder.compile()


def build_post_close_graph(context: TradingGraphContext):
    builder = StateGraph(TradingGraphState)

    def reconcile(state: TradingGraphState) -> dict:
        now = state["now"].astimezone(context.config.timezone)
        session_id = _session_id(now.date(), context.config.strategy_version)
        prices = _current_prices(context, state["now"])
        portfolio = context.portfolio_manager.evaluate(prices)
        report = {
            "trade_date": now.date().isoformat(),
            "portfolio": portfolio,
            "closed_at": now.isoformat(),
        }
        current = context.store.get_session(session_id)
        context.store.upsert_session(
            session_id, now.date().isoformat(), context.config.strategy_version,
            "CLOSED", buy_enabled=True,
            kill_switch=(current or {}).get("kill_switch", "NORMAL"),
            payload={**(current or {}).get("payload", {}), "close_report": report},
        )
        context.store.audit("POST_CLOSE", session_id, report)
        return {"session_id": session_id, "status": "CLOSED", "report": report}

    builder.add_node("reconcile", reconcile)
    builder.add_edge(START, "reconcile")
    builder.add_edge("reconcile", END)
    return builder.compile()
