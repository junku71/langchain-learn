from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from broker.base import Broker
from trading.display import stock_name
from trading.notifications import Notifier
from trading.models import ProtectionState
from trading.state_store import TradingStateStore


@dataclass
class ReconciliationResult:
    checked: int = 0
    changed: int = 0
    notified: int = 0
    errors: int = 0


class OrderReconciler:
    def __init__(self, broker: Broker, store: TradingStateStore, notifier: Notifier):
        self.broker = broker
        self.store = store
        self.notifier = notifier

    def reconcile(self) -> ReconciliationResult:
        summary = ReconciliationResult()
        self._import_daily_ledger(summary)
        for intent in self.store.list_reconcilable_orders():
            summary.checked += 1
            try:
                self._reconcile_one(intent, summary)
            except NotImplementedError:
                break
            except Exception as error:
                summary.errors += 1
                self.store.audit(
                    "ORDER_RECONCILE_FAILED",
                    intent["idempotency_key"],
                    {"error": f"{type(error).__name__}: {error}"},
                )
        return summary

    def _import_daily_ledger(self, summary: ReconciliationResult) -> None:
        """Import today's KIS ledger so external/already-filled orders are visible."""
        today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        try:
            executions = self.broker.list_order_executions(today)
        except NotImplementedError:
            return
        except Exception as error:
            summary.errors += 1
            self.store.audit(
                "DAILY_ORDER_IMPORT_FAILED", today,
                {"error": f"{type(error).__name__}: {error}"},
            )
            return
        for execution in executions:
            raw_date = execution.order_date.replace("-", "") or today.replace("-", "")
            raw_time = execution.order_time.zfill(6)
            try:
                local_time = datetime.strptime(
                    raw_date + raw_time, "%Y%m%d%H%M%S"
                ).replace(tzinfo=ZoneInfo("Asia/Seoul"))
                updated_at = local_time.astimezone(timezone.utc).isoformat()
            except ValueError:
                updated_at = datetime.now(timezone.utc).isoformat()
            if self.store.import_broker_execution(execution, updated_at):
                summary.changed += 1
                key = f"KIS:{raw_date}:{execution.order_id}"
                self._notify_execution(key, execution, summary)

    def _reconcile_one(self, intent: dict, summary: ReconciliationResult) -> None:
        order_date = intent["session_id"][:10]
        execution = self.broker.get_order_execution(
            intent["broker_order_id"], order_date, intent["ticker"]
        )
        if execution is None:
            return
        previous_status = intent["status"]
        previous_filled = int(intent["payload"].get("filled_quantity", 0) or 0)
        status_changed = execution.status != previous_status
        fill_changed = execution.filled_quantity != previous_filled
        if not status_changed and not fill_changed:
            return

        payload = {
            "execution": asdict(execution),
            "filled_quantity": execution.filled_quantity,
            "remaining_quantity": execution.remaining_quantity,
            "average_fill_price": execution.average_fill_price,
        }
        self.store.update_order_intent(
            intent["idempotency_key"],
            execution.status,
            broker_order_id=execution.order_id,
            payload=payload,
        )
        self.store.audit("ORDER_RECONCILED", intent["idempotency_key"], payload)
        summary.changed += 1

        if execution.side == "BUY" and execution.filled_quantity > 0:
            source = intent["payload"]
            highest = execution.average_fill_price or execution.order_price
            trailing_pct = source.get("trailing_stop_pct")
            self.store.save_protection(ProtectionState(
                ticker=execution.ticker,
                stop_loss=source.get("stop_loss"),
                take_profit=source.get("take_profit"),
                trailing_stop_pct=trailing_pct,
                trailing_stop=(
                    highest * (1 - float(trailing_pct))
                    if trailing_pct is not None else None
                ),
                highest_price=highest,
                updated_at=datetime.now(timezone.utc),
            ))

        should_notify = execution.filled_quantity > 0 or (
            status_changed and execution.status in {"REJECTED", "CANCELLED"}
        )
        if not should_notify:
            return
        self._notify_execution(intent["idempotency_key"], execution, summary)

    def _notify_execution(self, intent_key, execution, summary) -> None:
        should_notify = execution.filled_quantity > 0 or execution.status in {
            "REJECTED", "CANCELLED"
        }
        if not should_notify or not self.notifier.enabled:
            return
        event_key = f"{intent_key}:{execution.status}:{execution.filled_quantity}"
        message = self._fill_message(execution)
        if not self.store.begin_notification(event_key, "slack", {"text": message}):
            return
        try:
            self.notifier.send(message)
        except Exception as error:
            self.store.finish_notification(event_key, sent=False, error=str(error))
            self.store.audit(
                "NOTIFICATION_FAILED", event_key,
                {"channel": "slack", "error": f"{type(error).__name__}: {error}"},
            )
            summary.errors += 1
            return
        self.store.finish_notification(event_key, sent=True)
        self.store.audit("NOTIFICATION_SENT", event_key, {"channel": "slack"})
        summary.notified += 1

    @staticmethod
    def _fill_message(execution) -> str:
        if execution.status in {"REJECTED", "CANCELLED"}:
            label = "주문 거부" if execution.status == "REJECTED" else "주문 취소"
            name = execution.name or stock_name(execution.ticker)
            return (
                f":warning: *KIS {label}*\n"
                f"• 종목: {execution.ticker} {name}\n"
                f"• 구분: {execution.side}\n"
                f"• 주문수량: {execution.ordered_quantity:,}주\n"
                f"• 주문번호: `{execution.order_id}`"
            )
        emoji = ":large_green_circle:" if execution.side == "BUY" else ":red_circle:"
        label = "전량 체결" if execution.status == "FILLED" else "부분 체결"
        name = execution.name or stock_name(execution.ticker)
        price_label = (
            "실제 매도가격" if execution.side == "SELL" else "실제 매수가격"
        )
        fill_price = (
            f"{execution.average_fill_price:,.0f}원"
            if execution.average_fill_price > 0 else "조회되지 않음"
        )
        return (
            f"{emoji} *KIS {label}*\n"
            f"• 종목: {execution.ticker} {name}\n"
            f"• 구분: {execution.side}\n"
            f"• 체결: {execution.filled_quantity:,} / {execution.ordered_quantity:,}주\n"
            f"• {price_label}: {fill_price}\n"
            f"• 주문번호: `{execution.order_id}`"
        )
