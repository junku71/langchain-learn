from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from trading.protection import protection_from_position
from trading.trade_logging import log_trade_safely


def execute_due_scheduled_orders(context, now: datetime) -> dict:
    """Submit due approved rebalance reservations once, with SELLs first."""
    local = now.astimezone(context.config.timezone)
    if context.calendar.phase(local) != "REGULAR":
        return {"status": "MARKET_CLOSED", "orders": []}
    execute_on = local.date().isoformat()
    queued = context.store.list_scheduled_orders(statuses=("QUEUED",))
    due = [item for item in queued if item["execute_on"] <= execute_on]
    results = []
    session_id = f"{execute_on}:{context.config.strategy_version}"
    session = context.store.get_session(session_id)

    for item in due:
        reservation_id = item["reservation_id"]
        if not context.store.claim_scheduled_order(reservation_id, execute_on):
            continue
        try:
            if session and session.get("kill_switch") == "HALTED":
                raise RuntimeError("KILL_SWITCH_HALTED")
            ticker = item["ticker"]
            requested = int(item["quantity"])
            market_price = float(context.broker.get_current_price(ticker))
            price = float(item["payload"].get("limit_price") or market_price)
            if item["side"] == "SELL":
                position = context.broker.get_position(ticker)
                quantity = min(requested, position.quantity if position else 0)
                if quantity <= 0:
                    raise RuntimeError("NO_POSITION_TO_SELL")
            else:
                cash = float(context.broker.get_balance().get("cash", 0))
                quantity = min(requested, int(cash // price))
                if quantity <= 0:
                    raise RuntimeError("INSUFFICIENT_CASH")

            key = f"{session_id}:RESERVED:{reservation_id}"
            intent = {
                **item["payload"], "price": price, "quantity": quantity,
                "reservation_id": reservation_id,
            }
            if not context.store.create_order_intent(
                key, session_id, ticker, item["side"], intent
            ):
                raise RuntimeError("DUPLICATE_ORDER")
            if context.config.dry_run:
                result = {
                    "status": "DRY_RUN", "ticker": ticker,
                    "side": item["side"], "price": price,
                    "quantity": quantity, "order_id": "",
                    "reason": "SCHEDULED_REBALANCE",
                }
            elif item["side"] == "SELL":
                broker_result = context.broker.sell(
                    ticker, price, quantity,
                    order_type=item["payload"].get(
                        "order_type", "PRIORITY_LIMIT"
                    ),
                    reason="SCHEDULED_REBALANCE",
                )
                result = asdict(broker_result)
            else:
                broker_result = context.broker.buy(
                    ticker, price, quantity,
                    sector=item["payload"].get("sector", "UNKNOWN"),
                    stop_loss=item["payload"].get("stop_loss"),
                    take_profit=item["payload"].get("take_profit"),
                    trailing_stop_pct=(
                        item["payload"].get("trailing_stop_pct")
                        if item["payload"].get("trailing_stop_pct") is not None
                        else context.config.trailing_stop_pct
                    ),
                    order_type=item["payload"].get(
                        "order_type", "PRIORITY_LIMIT"
                    ),
                    reason="SCHEDULED_REBALANCE",
                )
                result = asdict(broker_result)
            context.store.update_order_intent(
                key, result["status"],
                broker_order_id=result.get("order_id", ""), payload=result,
            )
            context.store.finish_scheduled_order(
                reservation_id, "SUBMITTED",
                broker_order_id=result.get("order_id", ""), payload=result,
            )
            if not context.config.dry_run:
                log_trade_safely(context, broker_result, entity_key=key)
            position = context.broker.get_position(ticker)
            if item["side"] == "BUY" and position is not None and not context.config.dry_run:
                context.store.save_protection(protection_from_position(position, local))
            context.store.audit("SCHEDULED_ORDER_SUBMITTED", reservation_id, result)
            results.append({**result, "reservation_id": reservation_id})
        except Exception as error:
            failed = {"error": f"{type(error).__name__}: {error}"}
            context.store.finish_scheduled_order(reservation_id, "FAILED", payload=failed)
            context.store.audit("SCHEDULED_ORDER_FAILED", reservation_id, failed)
            results.append({**item, "status": "FAILED", **failed})
    return {"status": "COMPLETED", "orders": results}
