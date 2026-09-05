from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from broker.kis import KISBroker, KISConfig
from broker.models import OrderExecution, OrderResult
from broker.paper import PaperBroker
from paper.trade_logger import TradeLogger
from trading.calendar import KrxCalendar
from trading.config import LiveTradingConfig, SEOUL
from trading.console import TradingConsole
from trading.controller import TradingControlError, TradingController
from trading.graphs import TradingGraphContext
from trading.display import korean_time
from trading.models import MinuteBar, ProtectionState
from trading.notifications import SlackConfig, SlackNotifier
from trading.protection import evaluate_minute_bar
from trading.providers import KISMinuteBarProvider, PassthroughCandidateAnalysis
from trading.reconciliation import OrderReconciler, ReconciliationResult
from trading.rebalance import (
    MarketNewsService,
    RebalanceAction,
    RebalanceExecutor,
    RebalanceProposal,
    RebalanceValidator,
)
from analysis.news_models import NewsItem
from trading.recommendations import RecommendationService
from trading.service import LiveTradingService
from trading.state_store import TradingStateStore
from trading.web_app import order_rows


class FixedCandidates:
    def candidates(self, trade_date, per_market):
        return [{
            "ticker": "005930.KS",
            "name": "삼성전자",
            "market": "KOSPI",
            "sector": "반도체",
            "ml_score": 0.8,
            "classification_probability": 0.8,
            "ml_rank": 1,
            "atr_pct": 0.02,
        }]


class LowScoreCandidates:
    def candidates(self, trade_date, per_market):
        candidate = FixedCandidates().candidates(trade_date, per_market)[0]
        candidate["classification_probability"] = 0.10
        candidate["ml_rank"] = 99
        return [candidate]


class PaperQuotes:
    def __init__(self, broker):
        self.broker = broker

    def minute_bar(self, ticker, now):
        price = self.broker.get_current_price(ticker)
        return MinuteBar(ticker, now, price, price, price, price)


def _service(tmp_path, candidate_provider=None):
    broker = PaperBroker(initial_cash=1_000_000, commission_rate=0)
    broker.set_market_price("005930.KS", 100)
    config = replace(
        LiveTradingConfig(),
        dry_run=False,
        state_db_path=tmp_path / "trading.sqlite3",
        rebalance_report_dir=tmp_path / "reports",
        max_new_positions_per_day=1,
    )
    context = TradingGraphContext(
        config=config,
        calendar=KrxCalendar(),
        store=TradingStateStore(config.state_db_path),
        broker=broker,
        quote_provider=PaperQuotes(broker),
        candidate_provider=candidate_provider or FixedCandidates(),
        analysis=PassthroughCandidateAnalysis(),
        trade_logger=TradeLogger(str(tmp_path / "trades.csv")),
    )
    return LiveTradingService(context), broker, context.store


def test_opening_buy_then_minute_stop_sell(tmp_path):
    service, broker, store = _service(tmp_path)

    pre_open = service.run_due(datetime(2025, 1, 6, 8, 45, tzinfo=SEOUL))
    assert any(result.job == "pre_open" and result.status == "COMPLETED" for result in pre_open)

    opening = service.run_due(datetime(2025, 1, 6, 9, 5, tzinfo=SEOUL))
    assert any(result.job == "opening_buy" and result.status == "COMPLETED" for result in opening)
    position = broker.get_position("005930.KS")
    assert position is not None
    protection = store.get_protection("005930.KS")
    assert protection is not None
    assert protection.stop_loss == 96

    broker.set_market_price("005930.KS", 95)
    monitored = service.run_due(datetime(2025, 1, 6, 9, 6, tzinfo=SEOUL))
    monitor = next(result for result in monitored if result.job == "position_monitor")
    assert monitor.status == "COMPLETED"
    assert broker.get_position("005930.KS") is None
    assert store.get_protection("005930.KS") is None

    duplicate = service.run_monitor(datetime(2025, 1, 6, 9, 6, 30, tzinfo=SEOUL))
    assert duplicate.status == "DUPLICATE_SKIPPED"


def test_stop_and_target_same_minute_prefers_stop():
    protection = ProtectionState(
        ticker="005930.KS",
        stop_loss=95,
        take_profit=110,
        trailing_stop_pct=None,
        trailing_stop=None,
        highest_price=100,
        updated_at=datetime(2025, 1, 6, 9, 0, tzinfo=SEOUL),
    )
    bar = MinuteBar(
        "005930.KS",
        datetime(2025, 1, 6, 9, 1, tzinfo=SEOUL),
        100, 111, 94, 105,
    )

    action, price, _ = evaluate_minute_bar(protection, bar)

    assert action == "STOP_AND_TARGET_SAME_BAR"
    assert price == 95


def test_state_store_order_intent_is_idempotent(tmp_path):
    store = TradingStateStore(tmp_path / "state.sqlite3")
    first = store.create_order_intent(
        "2025-01-06:test:005930.KS:BUY",
        "2025-01-06:test",
        "005930.KS",
        "BUY",
        {"quantity": 1},
    )
    second = store.create_order_intent(
        "2025-01-06:test:005930.KS:BUY",
        "2025-01-06:test",
        "005930.KS",
        "BUY",
        {"quantity": 1},
    )

    assert first is True
    assert second is False


def test_console_controller_manual_buy_persists_protection(tmp_path):
    service, broker, store = _service(tmp_path)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)

    result = controller.manual_buy(
        "005930.KS",
        10,
        limit_price=100,
        sector="반도체",
        stop_loss=95,
        take_profit=110,
        trailing_stop_pct=0.08,
    )

    assert result.status == "FILLED"
    assert broker.get_position("005930.KS").quantity == 10
    protection = store.get_protection("005930.KS")
    assert protection is not None
    assert protection.stop_loss == 95
    assert protection.take_profit == 110
    assert protection.trailing_stop == 92
    assert store.list_order_intents(limit=1)[0]["status"] == "FILLED"


def test_manual_buy_resolves_name_and_is_queued_before_market_open(
    tmp_path, monkeypatch
):
    service, broker, store = _service(tmp_path)
    controller = TradingController(service)
    monkeypatch.setattr(
        "trading.controller.get_domestic_security",
        lambda name: {
            "name": "삼성전자", "ticker": "005930.KS", "market": "KOSPI"
        },
    )
    controller.now = lambda: datetime(2025, 1, 6, 8, 30, tzinfo=SEOUL)

    security = controller.resolve_domestic_security("삼성전자")
    result = controller.manual_buy_or_reserve(
        security["ticker"], 2, limit_price=100, sector=security["sector"],
        name=security["name"], stop_loss=95, take_profit=110,
        trailing_stop_pct=0.08, order_type="BEST_LIMIT",
    )

    assert security["sector"] == "전기·전자"
    assert result["status"] == "QUEUED"
    assert result["execute_on"] == "2025-01-06"
    assert broker.get_position("005930.KS") is None
    queued = store.list_scheduled_orders(statuses=("QUEUED",))[0]
    assert queued["payload"]["name"] == "삼성전자"
    assert queued["payload"]["stop_loss"] == 95
    assert queued["payload"]["order_type"] == "BEST_LIMIT"
    assert queued["payload"]["limit_price"] is None

    opening = service.run_scheduled_orders(
        datetime(2025, 1, 6, 9, 0, tzinfo=SEOUL)
    )

    assert opening.status == "COMPLETED"
    position = broker.get_position("005930.KS")
    assert position is not None
    assert position.quantity == 2
    assert position.sector == "전기·전자"
    assert position.stop_loss == 95
    assert position.take_profit == 110


def test_console_controller_controls_block_manual_buy(tmp_path):
    service, _, _ = _service(tmp_path)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    assert controller.toggle_kill_switch() == "HALTED"

    try:
        controller.manual_buy("005930.KS", 1, limit_price=100)
    except TradingControlError as error:
        assert "Kill Switch" in str(error)
    else:
        raise AssertionError("Kill Switch must block manual buys")


def test_controller_cancel_preserves_original_order_for_reconciliation(tmp_path):
    service, broker, store = _service(tmp_path)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    session_id = controller.current_session_id()
    key = f"{session_id}:manual:BUY:005930.KS:cancel-test"
    store.create_order_intent(
        key, session_id, "005930.KS", "BUY", {"quantity": 10, "price": 100}
    )
    store.update_order_intent(
        key, "PARTIALLY_FILLED", broker_order_id="12345",
        payload={"filled_quantity": 4, "remaining_quantity": 6},
    )
    broker.cancel_order = lambda *args, **kwargs: OrderResult(
        status="CANCEL_SUBMITTED", ticker="005930.KS", side="CANCEL",
        quantity=6, order_id="98765", reason="accepted",
    )

    outcome = controller.cancel_pending_order(key)

    assert outcome["request"]["status"] == "CANCEL_SUBMITTED"
    assert outcome["order"]["status"] == "PARTIALLY_FILLED"
    assert outcome["order"]["broker_order_id"] == "12345"
    assert outcome["order"]["payload"]["cancel_request"]["order_id"] == "98765"
    audit = store.list_audit_events(limit=1)[0]
    assert audit["event_type"] == "ORDER_CANCEL_REQUESTED"


def test_controller_api_reconciliation_runs_immediately_and_audits(tmp_path):
    service, _, store = _service(tmp_path)

    class Reconciler:
        def reconcile(self):
            return ReconciliationResult(checked=2, changed=1, notified=1, errors=0)

    service.reconciler = Reconciler()
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)

    report = controller.api_reconciliation()

    assert report == {"checked": 2, "changed": 1, "notified": 1, "errors": 0}
    audit = store.list_audit_events(limit=1)[0]
    assert audit["event_type"] == "MANUAL_RECONCILIATION"


def test_console_order_table_displays_price_times_quantity(tmp_path, capsys):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))

    console._order_table([{
        "updated_at": "2026-09-01T05:00:00+00:00",
        "ticker": "005930.KS", "side": "BUY", "status": "SUBMITTED",
        "broker_order_id": "12345", "payload": {"price": 70_000, "quantity": 3},
    }])

    output = capsys.readouterr().out
    assert "총주문금액" in output
    assert "210,000" in output


def test_recent_orders_reconciles_before_display(tmp_path, capsys):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))
    calls = []
    console.controller.api_reconciliation = lambda: (
        calls.append("reconcile")
        or {"checked": 2, "changed": 1, "notified": 0, "errors": 0}
    )
    console.controller.order_history = lambda: calls.append("history") or []

    console._orders()

    output = capsys.readouterr().out
    assert calls == ["reconcile", "history"]
    assert "[KIS 주문체결 reconciliation]" in output
    assert "[최근 주문 내역]" in output
    assert "2" in output


def test_recent_orders_still_displays_local_history_when_reconciliation_fails(
    tmp_path, capsys
):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))
    console.controller.api_reconciliation = lambda: (_ for _ in ()).throw(
        TradingControlError("KIS API 오류")
    )
    console.controller.order_history = lambda: []

    console._orders()

    output = capsys.readouterr().out
    assert "reconciliation을 완료하지 못했습니다" in output
    assert "[최근 주문 내역]" in output


def test_streamlit_order_rows_include_name_and_total_amount():
    rows = order_rows([{
        "updated_at": "2026-09-01T05:00:00+00:00",
        "ticker": "005930.KS", "side": "BUY", "status": "SUBMITTED",
        "broker_order_id": "12345", "idempotency_key": "key",
        "payload": {"price": 70_000, "quantity": 3},
    }])

    assert rows[0]["종목명"]
    assert rows[0]["총주문금액"] == 210_000


def test_ml_filter_on_rejects_and_off_routes_candidate_to_risk(tmp_path):
    on_service, on_broker, on_store = _service(
        tmp_path / "on", candidate_provider=LowScoreCandidates()
    )
    on_store.set_control("ml_filter_enabled", True)
    on_service.run_pre_open(datetime(2025, 1, 6, 8, 45, tzinfo=SEOUL))
    on_service.run_opening_buy(datetime(2025, 1, 6, 9, 5, tzinfo=SEOUL))
    assert on_broker.get_position("005930.KS") is None

    off_service, off_broker, off_store = _service(
        tmp_path / "off", candidate_provider=LowScoreCandidates()
    )
    off_store.set_control("ml_filter_enabled", False)
    off_service.run_pre_open(datetime(2025, 1, 6, 8, 45, tzinfo=SEOUL))
    result = off_service.run_opening_buy(
        datetime(2025, 1, 6, 9, 5, tzinfo=SEOUL)
    )

    assert result.status == "COMPLETED"
    assert off_broker.get_position("005930.KS") is not None
    assert off_store.get_bool_control("ml_filter_enabled", True) is False


class RecordingNotifier:
    enabled = True

    def __init__(self):
        self.messages = []

    def send(self, text, *, blocks=None):
        self.messages.append(text)


def test_order_reconciliation_updates_fill_and_notifies_once(tmp_path):
    broker = PaperBroker(initial_cash=1_000_000, commission_rate=0)
    broker.get_order_execution = lambda order_id, order_date, ticker=None: OrderExecution(
        order_id=order_id,
        ticker=ticker,
        side="BUY",
        status="FILLED",
        ordered_quantity=10,
        filled_quantity=10,
        remaining_quantity=0,
        average_fill_price=101,
        order_date=order_date,
        name="삼성전자",
    )
    store = TradingStateStore(tmp_path / "reconcile.sqlite3")
    key = "2025-01-06:test:manual:BUY:005930.KS"
    store.create_order_intent(
        key, "2025-01-06:test", "005930.KS", "BUY", {"quantity": 10}
    )
    store.update_order_intent(key, "SUBMITTED", broker_order_id="12345")
    notifier = RecordingNotifier()
    reconciler = OrderReconciler(broker, store, notifier)

    first = reconciler.reconcile()
    second = reconciler.reconcile()

    order = store.list_order_intents(limit=1)[0]
    assert first.changed == 1
    assert first.notified == 1
    assert second.checked == 0
    assert order["status"] == "FILLED"
    assert order["payload"]["filled_quantity"] == 10
    assert len(notifier.messages) == 1
    assert "실제 매수가격: 101원" in notifier.messages[0]


def test_reconciliation_imports_filled_order_missing_from_local_store(tmp_path):
    broker = PaperBroker(initial_cash=1_000_000, commission_rate=0)
    broker.list_order_executions = lambda order_date: [OrderExecution(
        order_id="98765", ticker="005930.KS", side="SELL", status="FILLED",
        ordered_quantity=3, filled_quantity=3, remaining_quantity=0,
        order_price=70_000, average_fill_price=70_100,
        order_date=order_date.replace("-", ""), order_time="101530",
        name="삼성전자",
    )]
    store = TradingStateStore(tmp_path / "daily-ledger.sqlite3")
    notifier = RecordingNotifier()
    reconciler = OrderReconciler(broker, store, notifier)

    first = reconciler.reconcile()
    second = reconciler.reconcile()

    orders = store.list_order_intents(limit=10)
    assert first.changed == 1
    assert first.notified == 1
    assert second.changed == 0
    assert second.notified == 0
    assert len(orders) == 1
    assert orders[0]["broker_order_id"] == "98765"
    assert orders[0]["status"] == "FILLED"
    assert orders[0]["payload"]["price"] == 70_100
    assert orders[0]["payload"]["source"] == "kis_daily_ledger"
    assert len(notifier.messages) == 1


def test_order_timestamp_is_displayed_in_korean_time():
    assert korean_time("2025-01-06T01:15:00+00:00") == "2025-01-06 10:15:00"


def test_slack_sell_fill_message_includes_actual_sale_price():
    execution = OrderExecution(
        order_id="12345",
        ticker="005930.KS",
        side="SELL",
        status="FILLED",
        ordered_quantity=10,
        filled_quantity=10,
        remaining_quantity=0,
        average_fill_price=71_250,
        name="삼성전자",
    )

    message = OrderReconciler._fill_message(execution)

    assert "실제 매도가격: 71,250원" in message


def test_kis_execution_response_is_normalized():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    captured = {}

    def fake_request(method, path, tr_id, **kwargs):
        captured.update({"method": method, "path": path, "tr_id": tr_id})
        return {"output1": [{
            "odno": "0000012345", "pdno": "005930", "prdt_name": "삼성전자",
            "sll_buy_dvsn_cd": "02", "ord_qty": "10", "tot_ccld_qty": "4",
            "rmn_qty": "6", "ord_unpr": "100", "avg_prvs": "101",
            "ord_dt": "20250106", "ord_tmd": "091500",
            "rjct_yn": "N", "cncl_yn": "N",
        }]}

    broker._request = fake_request
    execution = broker.get_order_execution("12345", "2025-01-06", "005930.KS")

    assert captured["tr_id"] == "VTTC0081R"
    assert captured["path"].endswith("inquire-daily-ccld")
    assert execution.status == "PARTIALLY_FILLED"
    assert execution.filled_quantity == 4
    assert execution.remaining_quantity == 6


def test_kis_cancel_order_submits_official_full_cancel_fields():
    broker = KISBroker(KISConfig(
        "key", "secret", "12345678", enable_trading=True
    ))
    broker.get_order_execution = lambda *args, **kwargs: OrderExecution(
        order_id="12345", ticker="005930.KS", side="BUY",
        status="PARTIALLY_FILLED", ordered_quantity=10,
        filled_quantity=4, remaining_quantity=6,
        raw={"ord_gno_brno": "00950", "ord_dvsn_cd": "00"},
    )
    captured = {}

    def fake_request(method, path, tr_id, **kwargs):
        captured.update({
            "method": method, "path": path, "tr_id": tr_id,
            "body": kwargs["body"],
        })
        return {"output": {"ODNO": "98765"}}

    broker._request = fake_request
    result = broker.cancel_order("12345", "2025-01-06", "005930.KS")

    assert result.status == "CANCEL_SUBMITTED"
    assert result.quantity == 6
    assert result.order_id == "98765"
    assert captured["tr_id"] == "VTTC0013U"
    assert captured["path"].endswith("order-rvsecncl")
    assert captured["body"]["KRX_FWDG_ORD_ORGNO"] == "00950"
    assert captured["body"]["ORGN_ODNO"] == "12345"
    assert captured["body"]["RVSE_CNCL_DVSN_CD"] == "02"
    assert captured["body"]["QTY_ALL_ORD_YN"] == "Y"
    assert captured["body"]["ORD_QTY"] == "0"


def test_kis_request_refreshes_server_expired_token_once():
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.ok = status_code < 400

        def json(self):
            return self.payload

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"HTTP {self.status_code}")

    class Session:
        def __init__(self):
            self.request_tokens = []
            self.token_requests = 0

        def post(self, url, **kwargs):
            self.token_requests += 1
            return Response(200, {"access_token": "fresh-token", "expires_in": 3600})

        def request(self, method, url, **kwargs):
            token = kwargs["headers"]["authorization"]
            self.request_tokens.append(token)
            if token == "Bearer stale-token":
                return Response(500, {
                    "rt_cd": "1", "msg_cd": "EGW00123",
                    "msg1": "기간이 만료된 token 입니다.",
                })
            return Response(200, {"rt_cd": "0", "output": {"ok": True}})

    session = Session()
    broker = KISBroker(KISConfig("key", "secret", "12345678"), session=session)
    broker._access_token = "stale-token"
    broker._token_expires_at = time.time() + 3600

    result = broker._request("GET", "/test", "TEST")

    assert result["output"]["ok"] is True
    assert session.request_tokens == ["Bearer stale-token", "Bearer fresh-token"]
    assert session.token_requests == 1


def test_kis_partially_filled_then_cancelled_is_terminal():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    broker._request = lambda *args, **kwargs: {"output1": [{
        "odno": "12345", "pdno": "005930", "sll_buy_dvsn_cd": "02",
        "ord_qty": "10", "tot_ccld_qty": "4", "rmn_qty": "6",
        "ord_unpr": "100", "avg_prvs": "101", "ord_dt": "20250106",
        "rjct_yn": "N", "cncl_yn": "Y",
    }]}

    execution = broker.get_order_execution("12345", "2025-01-06", "005930.KS")

    assert execution.status == "CANCELLED"
    assert execution.filled_quantity == 4


def test_kis_virtual_cancel_confirmation_does_not_resurrect_remaining_qty():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    broker._request = lambda *args, **kwargs: {"output1": [{
        "odno": "0000031756", "pdno": "003230",
        "sll_buy_dvsn_cd": "01", "ord_qty": "2",
        "tot_ccld_qty": "0", "cncl_cfrm_qty": "2", "rmn_qty": "0",
        "ord_unpr": "1509000", "ord_dt": "20260901",
        "rjct_yn": "N", "cncl_yn": "N",
    }]}

    execution = broker.get_order_execution(
        "0000031756", "2026-09-01", "003230.KS"
    )

    assert execution.status == "CANCELLED"
    assert execution.remaining_quantity == 0


def test_kis_execution_derives_actual_price_from_total_fill_amount():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    broker._request = lambda *args, **kwargs: {"output1": [{
        "odno": "12345", "pdno": "005930", "sll_buy_dvsn_cd": "01",
        "ord_qty": "10", "tot_ccld_qty": "10", "rmn_qty": "0",
        "ord_unpr": "71000", "tot_ccld_amt": "712500",
        "ord_dt": "20250106", "rjct_yn": "N", "cncl_yn": "N",
    }]}

    execution = broker.get_order_execution("12345", "2025-01-06", "005930.KS")

    assert execution.side == "SELL"
    assert execution.average_fill_price == 71_250


def test_kis_minute_response_is_normalized_and_completed_bar_selected():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    captured = {}

    def fake_request(method, path, tr_id, **kwargs):
        captured.update({
            "method": method, "path": path, "tr_id": tr_id,
            "params": kwargs["params"],
        })
        return {"output2": [
            {
                "stck_bsop_date": "20250106", "stck_cntg_hour": "090600",
                "stck_oprc": "103", "stck_hgpr": "104", "stck_lwpr": "102",
                "stck_prpr": "103", "cntg_vol": "11",
            },
            {
                "stck_bsop_date": "20250106", "stck_cntg_hour": "090500",
                "stck_oprc": "100", "stck_hgpr": "105", "stck_lwpr": "94",
                "stck_prpr": "102", "cntg_vol": "1234",
            },
        ]}

    broker._request = fake_request
    now = datetime(2025, 1, 6, 9, 6, 5, tzinfo=SEOUL)
    provider = KISMinuteBarProvider(broker)
    bar = provider.minute_bar("005930.KS", now)

    assert captured["tr_id"] == "FHKST03010230"
    assert captured["path"].endswith("inquire-time-dailychartprice")
    assert captured["params"]["FID_INPUT_DATE_1"] == "20250106"
    assert bar.timestamp == datetime(2025, 1, 6, 9, 5, tzinfo=SEOUL)
    assert (bar.open, bar.high, bar.low, bar.close) == (100, 105, 94, 102)
    assert bar.volume == 1234
    assert bar.source == "KIS_MINUTE"


def test_kis_minute_provider_falls_back_to_current_price():
    broker = KISBroker(KISConfig("key", "secret", "12345678"))
    broker.get_minute_bars = lambda ticker, now: []
    broker.get_current_price = lambda ticker: 101
    now = datetime(2025, 1, 6, 9, 6, 5, tzinfo=SEOUL)

    bar = KISMinuteBarProvider(broker).minute_bar("005930.KS", now)

    assert (bar.open, bar.high, bar.low, bar.close) == (101, 101, 101, 101)
    assert bar.source == "CURRENT_PRICE_FALLBACK"


def test_slack_webhook_posts_text_payload():
    calls = []

    class Response:
        text = "ok"

        @staticmethod
        def raise_for_status():
            return None

    class Session:
        @staticmethod
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    notifier = SlackNotifier(
        SlackConfig(
            enabled=True,
            webhook_url="https://hooks.slack.com/services/T/B/secret",
        ),
        session=Session(),
    )
    notifier.send("체결 테스트")

    assert calls[0][0].startswith("https://hooks.slack.com/")
    assert calls[0][1]["json"] == {"text": "체결 테스트"}


def test_slack_bot_uploads_html_report_and_returns_permalink(tmp_path):
    class Response:
        def __init__(self, payload=None):
            self.payload = payload or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def get(self, url, **kwargs):
            if url.endswith("files.getUploadURLExternal"):
                return Response({
                    "ok": True, "upload_url": "https://files.slack.test/upload",
                    "file_id": "F123",
                })
            return Response({
                "ok": True, "file": {"permalink": "https://workspace.slack.com/files/F123"},
            })

        def post(self, url, **kwargs):
            if url == "https://files.slack.test/upload":
                return Response()
            return Response({"ok": True, "files": [{"id": "F123"}]})

    report = tmp_path / "report.html"
    report.write_text("<html>report</html>", encoding="utf-8")
    notifier = SlackNotifier(
        SlackConfig(enabled=True, bot_token="xoxb-test", channel="C123"),
        session=Session(),
    )

    link = notifier.upload_file(report, title="리밸런싱 리포트")

    assert link == "https://workspace.slack.com/files/F123"


def test_top_recommendations_rank_and_cache_results(tmp_path):
    class Candidates:
        def candidates(self, trade_date, per_market):
            return [
                {"ticker": "005930.KS", "name": "삼성전자", "market": "KOSPI", "sector": "반도체", "ml_score": 0.8},
                {"ticker": "000660.KS", "name": "SK하이닉스", "market": "KOSPI", "sector": "반도체", "ml_score": 0.7},
            ]

    class News:
        calls = 0

        def collect(self, ticker):
            self.calls += 1
            title = "성장 호실적 상향" if ticker == "005930.KS" else "부진 하락 우려"
            return {
                "naver_news": [{"title": title, "description": ""}],
                "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    service, _, store = _service(tmp_path, candidate_provider=Candidates())
    store.set_control("ml_filter_enabled", True)
    news = News()
    recommendation = RecommendationService(
        service.context,
        technical_fn=lambda ticker: {
            "score": 90 if ticker == "005930.KS" else 40,
            "signal": "BULLISH" if ticker == "005930.KS" else "NEUTRAL",
            "indicators": {"RSI": 60},
        },
        fundamental_fn=lambda ticker: {
            "score": 80 if ticker == "005930.KS" else 50,
            "signal": "BULLISH", "PER": 12, "PBR": 1.2, "ROE": 15,
        },
        flow_fn=lambda ticker: {
            "score": 75 if ticker == "005930.KS" else 45,
            "signal": "BULLISH", "joint_buy_days": 3,
            "foreign_net_sum": 1000, "institution_net_sum": 500,
        },
        news_service=news,
    )

    first = recommendation.top_recommendations(datetime(2025, 1, 6).date())
    second = recommendation.top_recommendations(datetime(2025, 1, 6).date())
    next_day = recommendation.top_recommendations(datetime(2025, 1, 7).date())

    assert first[0]["ticker"] == "005930.KS"
    assert first[0]["rank"] == 1
    assert first[0]["recommendation_reason"]
    assert first == second
    assert next_day
    assert news.calls == 4


def test_market_news_only_keeps_articles_from_the_last_seven_days():
    now = datetime.now(timezone.utc)

    class NewsProvider:
        @staticmethod
        def search_query(query, display):
            return [
                NewsItem("test", "최근 시장 뉴스", "recent", (now - timedelta(days=2)).isoformat()),
                NewsItem("test", "오래된 시장 뉴스", "old", (now - timedelta(days=8)).isoformat()),
                NewsItem("test", "날짜 없는 시장 뉴스", "undated", None),
            ]

    result = MarketNewsService(provider=NewsProvider()).collect()

    assert [item["title"] for item in result["headlines"]] == ["최근 시장 뉴스"]
    assert result["news_since"]


def test_security_news_summary_is_link_free_and_at_most_three_short_lines():
    headlines = [
        {"title": f"뉴스 {index}", "description": "핵심 내용", "link": f"https://example.com/{index}"}
        for index in range(4)
    ]

    summary = MarketNewsService._security_news_summary(headlines)

    assert len(summary.splitlines()) == 3
    assert "https://" not in summary


def test_top_recommendations_ml_off_uses_market_cap_universe(tmp_path):
    class BrokenModelCandidates:
        def candidates(self, trade_date, per_market):
            raise ValueError("Model version is stale; retrain before live trading")

    class EmptyNews:
        def collect(self, ticker):
            return {
                "naver_news": [], "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    universe_path = tmp_path / "universe.csv"
    universe_path.write_text(
        "market,ticker,name,market_cap_rank,sector\n"
        "KOSPI,005930.KS,삼성전자,1,반도체\n"
        "KOSDAQ,247540.KQ,에코프로비엠,1,화학\n",
        encoding="utf-8",
    )
    service, _, store = _service(
        tmp_path / "service", candidate_provider=BrokenModelCandidates()
    )
    store.set_control("ml_filter_enabled", False)
    recommendation = RecommendationService(
        service.context,
        technical_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL", "indicators": {"RSI": 50}},
        fundamental_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        flow_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        news_service=EmptyNews(),
        universe_path=universe_path,
    )

    results = recommendation.top_recommendations(datetime(2025, 1, 6).date())

    assert {item["ticker"] for item in results} == {"005930.KS", "247540.KQ"}
    assert all(item["candidate_source"] == "MARKET_CAP_UNIVERSE" for item in results)


def test_top_recommendations_market_scope_filters_and_separates_cache(tmp_path):
    class Candidates:
        def candidates(self, trade_date, per_market):
            return [
                {"ticker": "005930.KS", "name": "삼성전자", "market": "KOSPI", "sector": "반도체", "ml_score": 0.8},
                {"ticker": "247540.KQ", "name": "에코프로비엠", "market": "KOSDAQ", "sector": "화학", "ml_score": 0.7},
            ]

    class EmptyNews:
        def collect(self, ticker):
            return {
                "naver_news": [], "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    service, _, store = _service(tmp_path, candidate_provider=Candidates())
    store.set_control("ml_filter_enabled", True)
    recommendation = RecommendationService(
        service.context,
        technical_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL", "indicators": {"RSI": 50}},
        fundamental_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        flow_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        news_service=EmptyNews(),
    )
    trade_date = datetime(2025, 1, 6).date()

    kospi = recommendation.top_recommendations(trade_date, universe_scope="KOSPI")
    kosdaq = recommendation.top_recommendations(trade_date, universe_scope="KOSDAQ")
    both = recommendation.top_recommendations(trade_date, universe_scope="BOTH")

    assert [item["ticker"] for item in kospi] == ["005930.KS"]
    assert [item["ticker"] for item in kosdaq] == ["247540.KQ"]
    assert {item["ticker"] for item in both} == {"005930.KS", "247540.KQ"}
    assert recommendation.top_recommendations(
        trade_date, universe_scope="KOSPI"
    ) == kospi


def test_top_recommendations_refresh_ignores_stale_session_candidates(tmp_path):
    class FreshCandidates:
        def candidates(self, trade_date, per_market):
            return [
                {"ticker": "005930.KS", "name": "삼성전자", "market": "KOSPI", "sector": "반도체", "ml_score": 0.99, "classification_probability": 0.99, "ml_rank": 1},
                {"ticker": "000660.KS", "name": "SK하이닉스", "market": "KOSPI", "sector": "반도체", "ml_score": 0.95, "classification_probability": 0.95, "ml_rank": 2},
            ]

    class EmptyNews:
        def collect(self, ticker):
            return {
                "naver_news": [], "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    service, _, store = _service(tmp_path, candidate_provider=FreshCandidates())
    trade_date = datetime(2025, 1, 6).date()
    session_id = f"{trade_date.isoformat()}:{service.context.config.strategy_version}"
    store.upsert_session(
        session_id,
        trade_date.isoformat(),
        service.context.config.strategy_version,
        "SESSION_READY",
        buy_enabled=True,
        payload={"candidates": [{"ticker": "OLD1.KS", "market": "KOSPI", "ml_score": 0.1, "classification_probability": 0.1, "ml_rank": 1}]},
    )

    recommendation = RecommendationService(
        service.context,
        technical_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL", "indicators": {"RSI": 50}},
        fundamental_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        flow_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        news_service=EmptyNews(),
    )

    results = recommendation.top_recommendations(
        trade_date, universe_scope="KOSPI", refresh=True
    )

    assert {item["ticker"] for item in results} == {"005930.KS", "000660.KS"}
    assert all(item["ticker"] != "OLD1.KS" for item in results)


def test_top_recommendations_analyzes_universe_then_ml_filters_top30(tmp_path):
    class Candidates:
        def candidates(self, trade_date, per_market):
            assert per_market == 100
            return [
                {
                    "ticker": f"{index:06d}.KS",
                    "name": f"stock-{index}",
                    "market": "KOSPI",
                    "sector": "TEST",
                    # Reverse the factor order so the test can prove that ML is
                    # applied only after the factor top-30 has been formed.
                    "ml_score": index / 100,
                    "classification_probability": index / 100,
                    "ml_rank": 40 - index,
                }
                for index in range(1, 41)
            ]

    class EmptyNews:
        def collect(self, ticker):
            return {
                "naver_news": [], "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    service, _, store = _service(tmp_path, candidate_provider=Candidates())
    store.set_control("ml_filter_enabled", True)
    analyzed = []

    def technical(ticker):
        index = int(ticker.split(".", 1)[0])
        analyzed.append(ticker)
        return {"score": 100 - index, "signal": "NEUTRAL", "indicators": {"RSI": 50}}

    recommendation = RecommendationService(
        service.context,
        technical_fn=technical,
        fundamental_fn=lambda ticker: {"score": 50, "signal": "NEUTRAL"},
        flow_fn=lambda ticker: {"score": 50, "signal": "NEUTRAL"},
        news_service=EmptyNews(),
    )

    results = recommendation.top_recommendations(
        datetime(2025, 1, 6).date(), universe_scope="KOSPI", refresh=True
    )

    assert len(analyzed) == 40
    assert len(results) == 10
    assert [item["ticker"] for item in results] == [
        f"{index:06d}.KS" for index in range(30, 20, -1)
    ]


def test_technical_recommendation_reason_contains_detailed_indicators():
    reason = RecommendationService._technical_reason({
        "signal": "BULLISH",
        "indicators": {
            "RSI": 61.2,
            "ADX": 27.4,
            "DI_PLUS": 31.5,
            "DI_MINUS": 18.1,
            "MACD": 125.6789,
            "MACD_SIGNAL": 110.1234,
            "volume_ratio": 1.82,
            "MA5": 71000,
            "MA20": 69000,
            "MA60": 65000,
            "ATR": 1350.5,
        },
    })

    assert "RSI 61.2" in reason
    assert "ADX 27.4" in reason
    assert "DI+ 31.5" in reason
    assert "DI- 18.1" in reason
    assert "MACD 125.679" in reason
    assert "Volume Ratio 1.82" in reason


def test_kr_top_recommendations_ignore_legacy_disabled_ml_control(tmp_path):
    class Candidates:
        def candidates(self, trade_date, per_market):
            return [{
                "ticker": "005930.KS",
                "name": "삼성전자",
                "market": "KOSPI",
                "sector": "전기·전자",
                "ml_score": 0.8123,
                "classification_probability": 0.77,
                "ml_rank": 1,
            }]

    class EmptyNews:
        def collect(self, ticker):
            return {
                "naver_news": [], "naver_earnings_news": [], "yahoo_news": [],
                "earnings_features": {},
            }

    service, _, store = _service(tmp_path, candidate_provider=Candidates())
    store.set_control("ml_filter_enabled", False)
    recommendation = RecommendationService(
        service.context,
        technical_fn=lambda ticker: {
            "score": 60, "signal": "NEUTRAL", "indicators": {"RSI": 50}
        },
        fundamental_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        flow_fn=lambda ticker: {"score": 60, "signal": "NEUTRAL"},
        news_service=EmptyNews(),
    )

    results = recommendation.top_recommendations(
        datetime(2025, 1, 6).date(), universe_scope="KOSPI", refresh=True
    )

    assert results[0]["ml_score"] == 0.8123
    assert results[0]["classification_probability"] == 0.77
    event = store.list_audit_events(limit=1)[0]
    assert event["payload"]["candidate_source"] == "ML_SNAPSHOT"
    assert event["payload"]["ml_filter_enabled"] is True


def test_console_top10_passes_selected_market_scope(tmp_path, monkeypatch, capsys):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))
    captured = {}

    def recommendations(**kwargs):
        captured.update(kwargs)
        return []

    console.controller.top_recommendations = recommendations
    answers = iter(["2", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    console._top_recommendations()

    assert captured["universe_scope"] == "KOSDAQ"
    assert captured["refresh"] is False
    assert "추천할 종목이 없습니다" in capsys.readouterr().out


def test_controller_saves_top10_html_and_slack_permalink_for_rebalancing(tmp_path):
    recommendation = {
        "rank": 1, "ticker": "005930.KS", "name": "삼성전자",
        "market": "KOSPI", "sector": "반도체", "total_score": 88,
        "fundamental_score": 81, "fundamental_reason": "PER과 ROE 양호",
        "technical_score": 92, "technical_reason": "상승 추세",
        "flow_score": 90, "flow_reason": "외국인·기관 동반매수",
        "news_score": 86, "news_reason": "실적 전망 개선",
        "recommendation_reason": "기술 및 수급 우위",
    }

    class Recommendations:
        @staticmethod
        def top_recommendations(*args, **kwargs):
            return [recommendation]

    class UploadNotifier:
        enabled = True

        def __init__(self):
            self.messages = []
            self.uploaded = []

        def upload_file(self, path, **kwargs):
            self.uploaded.append(path)
            return "https://slack.test/top10-report"

        def send(self, text, *, blocks=None):
            self.messages.append(text)

    service, _, store = _service(tmp_path)
    notifier = UploadNotifier()
    service.reconciler = type("Reconciler", (), {"notifier": notifier})()
    controller = TradingController(service, recommendation_service=Recommendations())
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)

    result = controller.top_recommendations(universe_scope="KOSPI")
    saved = store.get_control("top_recommendations_latest")
    report = Path(saved["report_path"])
    assert result == [recommendation]
    assert report.name == "Top10_Pick_20250106_KOSPI.pdf"
    assert report.read_bytes().startswith(b"%PDF-")
    assert saved["report_url"] == service.context.config.rebalance_report_base_url
    assert notifier.uploaded == []
    assert service.context.config.rebalance_report_base_url in notifier.messages[-1]
    assert report.name in notifier.messages[-1]
    assert controller.latest_top_recommendations() == ("KOSPI", [recommendation])


def test_ml_off_market_cap_universe_uses_top_100_per_market(tmp_path):
    rows = ["market,ticker,name,market_cap_rank,sector"]
    for market, suffix, offset in (("KOSPI", "KS", 0), ("KOSDAQ", "KQ", 500000)):
        for rank in range(1, 102):
            code = f"{offset + rank:06d}"
            rows.append(f"{market},{code}.{suffix},{market}{rank},{rank},TEST")
    universe_path = tmp_path / "top200_source.csv"
    universe_path.write_text("\n".join(rows), encoding="utf-8")
    service, _, store = _service(tmp_path / "service")
    store.set_control("ml_filter_enabled", False)
    recommendation = RecommendationService(
        service.context, universe_path=universe_path
    )

    candidates = recommendation._market_cap_candidates()

    assert len(candidates) == 200
    assert sum(item["market"] == "KOSPI" for item in candidates) == 100
    assert sum(item["market"] == "KOSDAQ" for item in candidates) == 100
    assert max(item["market_cap_rank"] for item in candidates) == 100


def test_bulk_protection_uses_average_price_and_three_atr(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy(
        "005930.KS", 100, 10, stop_loss=90, take_profit=115,
        trailing_stop_pct=0.12,
    )
    broker.set_market_price("005930.KS", 110)
    controller = TradingController(service, atr_provider=lambda ticker: 5)

    preview = controller.preview_bulk_atr_protection()

    assert store.get_protection("005930.KS") is None
    assert preview[0]["avg_price"] == 100
    assert preview[0]["current_price"] == 110
    assert preview[0]["stop_loss"] == 85
    assert preview[0]["take_profit"] == 120
    assert preview[0]["trailing_stop_pct"] == 0.08
    assert preview[0]["trailing_stop"] == 101.2

    results = controller.apply_bulk_atr_protection(preview)
    protection = store.get_protection("005930.KS")

    assert results[0]["status"] == "SAVED"
    assert protection.stop_loss == 85
    assert protection.take_profit == 120
    assert protection.trailing_stop_pct == 0.08
    assert protection.trailing_stop == 101.2


def test_positions_display_includes_profit_and_portfolio_totals(tmp_path, capsys):
    service, broker, _ = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    console = TradingConsole(TradingController(service))

    console._positions()

    output = capsys.readouterr().out
    assert "현재가" in output
    assert "종목별 총구매가" in output
    assert "종목별 총현재가" in output
    assert "수익률" in output
    assert "손익금액" in output
    assert "섹터" not in output
    assert "10.00%" in output
    assert "총구입금액" in output
    assert "현재 총주식금액" in output
    assert "1,000" in output
    assert "1,100" in output


def test_account_overview_concatenates_balance_and_positions(tmp_path, capsys):
    service, broker, _ = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    console = TradingConsole(TradingController(service))

    console._account_overview()

    output = capsys.readouterr().out
    assert "[잔고 요약]" in output
    assert "[보유종목 및 보호가격]" in output
    assert output.index("[잔고 요약]") < output.index("[보유종목 및 보호가격]")
    balance_section = output[:output.index("[보유종목 및 보호가격]")]
    expected_headers = [
        "총평가금액", "주식평가금액", "주식매입금액", "예수금",
        "D+2예수금", "실현손익", "평가손익",
    ]
    for header in expected_headers:
        assert header in balance_section
    assert "보유종목 수" not in balance_section


def test_console_menu_groups_account_settings_and_reconciliation(capsys):
    TradingConsole._menu()

    output = capsys.readouterr().out
    assert "[조회]" not in output
    assert "9. 잔고 및 보유종목 조회" in output
    assert "4. 미체결 주문 조회 및 취소" in output
    assert "1. Buy/Sell 예약종목 조회 및 취소/정정" in output
    assert "5. 최근주문체결내역" in output
    assert "6. 매도전략 선택(샹들리에|돈치안|직접지정)" in output
    assert "10. 오늘의 Top10 추천 및 결과저장" in output
    assert "11. 보유 및 Top10추천종목 리밸런싱 제안 및 승인 후 주문예약" in output
    assert "[설정관리]" in output
    assert output.index("[주문]") < output.index("[자동매매]")
    assert output.index("[자동매매]") < output.index("8. API reconciliation")
    assert output.index("[포트폴리오 관리]") < output.index("8. API reconciliation")
    settings = output[output.index("[설정관리]"):]
    for menu_number in ("12.", "13.", "14.", "15.", "16.", "17."):
        assert menu_number in settings


def test_protection_menu_displays_current_price_and_return(
    tmp_path, capsys, monkeypatch
):
    service, broker, _ = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    console = TradingConsole(TradingController(service))
    answers = iter(["1", "-", "-", "-"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    console._protection()

    output = capsys.readouterr().out
    assert "평균단가 100원 / 현재가 110원 / 손익률 10.00%" in output


def test_console_merged_submenus_route_to_selected_action(tmp_path, monkeypatch):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))
    called = []
    console._sell = lambda: called.append("partial_sell")
    console._sell_all = lambda: called.append("full_sell")

    answers = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    console._sell_menu()

    assert called == ["full_sell"]


def test_chandelier_exit_strategy_is_saved_and_trails_by_atr(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    controller = TradingController(service, atr_provider=lambda ticker: 5)

    preview = controller.preview_exit_strategy(
        "005930.KS", "CHANDELIER_EXIT", atr_multiple=3
    )
    state = controller.apply_exit_strategy(preview)

    assert preview["strategy_name"] == "샹들리에 Exit"
    assert preview["atr"] == 5
    assert preview["stop_loss"] == 85
    assert preview["take_profit"] is None
    assert preview["trailing_stop"] == 95
    assert store.get_protection("005930.KS").strategy == "CHANDELIER_EXIT"

    bar = MinuteBar(
        ticker="005930.KS", timestamp=datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL),
        open=110, high=120, low=109, close=119, volume=100, source="test",
    )
    action, _, updated = evaluate_minute_bar(state, bar)

    assert action == "HOLD"
    assert updated.trailing_stop == 105


def test_donchian_exit_strategy_uses_lower_channel_and_has_no_target(
    tmp_path, monkeypatch
):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    controller = TradingController(service, atr_provider=lambda ticker: 5)
    monkeypatch.setattr(controller, "_latest_donchian_low", lambda ticker, period: 92)

    preview = controller.preview_exit_strategy(
        "005930.KS", "DONCHIAN_TREND", donchian_period=20
    )
    controller.apply_exit_strategy(preview)
    saved = store.get_protection("005930.KS")

    assert preview["strategy_name"] == "돈치안 추세추종(20일)"
    assert preview["atr"] == 5
    assert saved.stop_loss == 92
    assert saved.trailing_stop == 92
    assert saved.take_profit is None
    assert saved.strategy == "DONCHIAN_TREND"
    assert saved.donchian_period == 20


def test_donchian_channel_is_refreshed_once_per_trading_day(tmp_path, monkeypatch):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    store.save_protection(ProtectionState(
        ticker="005930.KS", stop_loss=90, take_profit=None,
        trailing_stop_pct=None, trailing_stop=90, highest_price=110,
        updated_at=datetime(2025, 1, 3, 15, 0, tzinfo=SEOUL),
        strategy="DONCHIAN_TREND", atr=5, atr_multiple=3,
        donchian_period=20,
    ))
    calls = []
    monkeypatch.setattr(
        "trading.graphs.latest_donchian_low",
        lambda ticker, period: calls.append((ticker, period)) or 95,
    )

    first = service.run_monitor(datetime(2025, 1, 6, 9, 5, tzinfo=SEOUL))
    second = service.run_monitor(datetime(2025, 1, 6, 9, 6, tzinfo=SEOUL))

    assert first.status == "COMPLETED"
    assert second.status == "COMPLETED"
    assert calls == [("005930.KS", 20)]
    assert store.get_protection("005930.KS").trailing_stop == 95


def test_exit_strategy_menu_displays_calculated_values(tmp_path, monkeypatch, capsys):
    service, broker, _ = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    console = TradingConsole(
        TradingController(service, atr_provider=lambda ticker: 5)
    )
    answers = iter(["1", "1", "1", "", "APPLY SELECTED"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    console._protection_menu()

    output = capsys.readouterr().out
    assert "현재 설정된 매도전략" in output
    assert "매도전략명" in output
    assert "현재가격" in output
    assert "수익률" in output
    assert "샹들리에 Exit" in output
    assert "ATR" in output
    assert "손절가" in output
    assert "익절가" in output
    assert "Trailing Stop" in output


def test_exit_strategy_overview_defaults_missing_strategy_to_chandelier(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    controller = TradingController(service, atr_provider=lambda ticker: 5)

    rows = controller.exit_strategy_overview()

    assert len(rows) == 1
    assert rows[0]["name"]
    assert rows[0]["quantity"] == 10
    assert rows[0]["avg_price"] == 100
    assert rows[0]["current_price"] == 110
    assert rows[0]["return_pct"] == 0.10
    assert rows[0]["total_value"] == 1_000
    assert rows[0]["strategy_name"] == "샹들리에 Exit"
    assert rows[0]["atr"] == 5
    assert rows[0]["take_profit"] is None
    assert rows[0]["stop_loss"] == 85
    assert rows[0]["trailing_stop"] == 95
    assert store.get_protection("005930.KS").strategy == "CHANDELIER_EXIT"


def test_exit_strategy_menu_accepts_comma_separated_positions(
    tmp_path, monkeypatch, capsys
):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 2)
    broker.set_market_price("000660.KS", 200)
    broker.buy("000660.KS", 200, 3)
    console = TradingConsole(
        TradingController(service, atr_provider=lambda ticker: 5)
    )
    answers = iter([
        "1", "1, 2", "3", "", "", "", "APPLY SELECTED",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    console._protection_menu()

    assert store.get_protection("005930.KS").strategy == "DIRECT_SPECIFIED"
    assert store.get_protection("000660.KS").strategy == "DIRECT_SPECIFIED"
    output = capsys.readouterr().out
    assert "적용 완료(선택 종목): 저장 2개 / 실패 0개" in output


def test_bulk_exit_strategy_applies_to_every_holding(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 2)
    broker.set_market_price("000660.KS", 200)
    broker.buy("000660.KS", 200, 3)
    controller = TradingController(service, atr_provider=lambda ticker: 5)

    previews = controller.preview_bulk_exit_strategy(
        "CHANDELIER_EXIT", atr_multiple=2
    )
    results = controller.apply_bulk_exit_strategy(previews)

    assert len(results) == 2
    assert all(item["status"] == "SAVED" for item in results)
    assert store.get_protection("005930.KS").atr_multiple == 2
    assert store.get_protection("000660.KS").atr_multiple == 2


def test_direct_specified_exit_strategy_uses_default_percentages(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    broker.set_market_price("005930.KS", 110)
    controller = TradingController(service, atr_provider=lambda ticker: 5)

    preview = controller.preview_exit_strategy(
        "005930.KS", "DIRECT_SPECIFIED"
    )
    state = controller.apply_exit_strategy(preview)

    assert preview["strategy_name"] == (
        "직접지정(익절 +20%, 손절 -10%, Trailing 8%)"
    )
    assert preview["atr"] == 5
    assert preview["take_profit"] == 120
    assert preview["stop_loss"] == 90
    assert preview["trailing_stop"] == 101.2
    assert state.strategy == "DIRECT_SPECIFIED"
    assert state.trailing_stop_pct == 0.08
    assert store.get_protection("005930.KS").take_profit == 120


def test_sell_position_table_shows_total_value_and_price_method(tmp_path, monkeypatch, capsys):
    service, broker, _ = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    console = TradingConsole(TradingController(service))
    answers = iter(["1", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    ticker, position = console._select_position()
    order_type, price = console._sell_price_method(110)

    output = capsys.readouterr().out
    assert ticker == "005930.KS"
    assert position.quantity == 10
    assert "총평가금액" in output
    assert "1,000" in output
    assert "최우선지정가 (가격우선)" in output
    assert order_type == "PRIORITY_LIMIT"
    assert price == 110


def test_buy_price_method_supports_best_limit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "4")

    order_type, price = TradingConsole._buy_price_method(70000)

    output = capsys.readouterr().out
    assert "최유리지정가 (타이밍우선)" in output
    assert order_type == "BEST_LIMIT"
    assert price == 70000


def test_console_scheduler_menu_toggles_start_and_stop(tmp_path, monkeypatch):
    service, _, _ = _service(tmp_path)
    controller = TradingController(service)
    console = TradingConsole(controller)
    state = {"running": False}
    monkeypatch.setattr(
        TradingController, "scheduler_running",
        property(lambda self: state["running"]),
    )
    monkeypatch.setattr(
        controller, "start_scheduler",
        lambda: state.update(running=True) is None,
    )
    monkeypatch.setattr(
        controller, "stop_scheduler",
        lambda: state.update(running=False) is None,
    )

    console._scheduler_toggle()
    assert state["running"] is True
    console._scheduler_toggle()
    assert state["running"] is False


def test_rebalance_validator_converts_target_weights_to_safe_orders():
    config = replace(
        LiveTradingConfig(), rebalance_enabled=True,
        rebalance_max_turnover_pct=0.30,
    )
    snapshot = {
        "portfolio": {"total_equity": 1_000_000, "cash": 500_000},
        "positions": [{
            "ticker": "005930.KS", "name": "삼성전자", "quantity": 100,
            "current_price": 1_000, "weight_pct": 10, "sector": "반도체",
        }],
        "top10": [{
            "ticker": "000660.KS", "name": "SK하이닉스",
            "current_price": 2_000, "sector": "반도체",
        }],
    }
    proposal = RebalanceProposal(
        market_view="NEUTRAL", market_summary="테스트",
        recommended_cash_pct=40, overall_reason="테스트",
        actions=[
            RebalanceAction(
                ticker="005930.KS", action="REDUCE", target_weight_pct=5,
                confidence=0.9, reason="비중 축소",
            ),
            RebalanceAction(
                ticker="000660.KS", action="BUY", target_weight_pct=10,
                confidence=0.9, reason="Top10 편입",
            ),
        ],
    )

    result = RebalanceValidator(config).validate(snapshot, proposal)

    assert result["approved"] is True
    assert [(item["side"], item["quantity"]) for item in result["orders"]] == [
        ("SELL", 50), ("BUY", 50),
    ]


def test_rebalance_validator_rejects_new_buy_outside_top10():
    snapshot = {
        "portfolio": {"total_equity": 1_000_000, "cash": 1_000_000},
        "positions": [], "top10": [],
    }
    proposal = RebalanceProposal(
        market_view="RISK_ON", market_summary="테스트",
        recommended_cash_pct=80, overall_reason="테스트",
        actions=[RebalanceAction(
            ticker="123456.KS", action="BUY", target_weight_pct=10,
            confidence=0.9, reason="근거 없음",
        )],
    )

    result = RebalanceValidator(LiveTradingConfig()).validate(snapshot, proposal)

    assert result["approved"] is False
    assert result["override_allowed"] is False
    assert "Top10 외 신규 매수" in result["errors"][0]


def test_rebalance_cash_policy_rejection_can_be_overridden():
    config = replace(LiveTradingConfig(), rebalance_min_cash_pct=0.10)
    snapshot = {
        "portfolio": {"total_equity": 1_000_000, "cash": 500_000},
        "positions": [],
        "top10": [{
            "ticker": "000660.KS", "name": "SK하이닉스",
            "current_price": 2_000, "sector": "반도체",
        }],
    }
    proposal = RebalanceProposal(
        market_view="NEUTRAL", market_summary="테스트",
        recommended_cash_pct=45, overall_reason="테스트",
        actions=[RebalanceAction(
            ticker="000660.KS", action="BUY", target_weight_pct=10,
            confidence=0.9, reason="Top10 편입",
        )],
    )

    result = RebalanceValidator(config).validate(snapshot, proposal)

    assert result["approved"] is False
    assert result["override_allowed"] is True
    assert result["hard_errors"] == []
    assert "예상 현금 비중" in result["errors"][0]


def test_controller_records_explicit_rebalance_risk_override(tmp_path, monkeypatch):
    service, _, store = _service(tmp_path)
    service.context.config = replace(service.context.config, rebalance_enabled=True)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    package = {
        "proposal_id": "RB-TEST-OVERRIDE",
        "created_at": controller.now().isoformat(),
        "validation": {
            "approved": False,
            "override_allowed": True,
            "errors": ["예상 현금 비중이 요구 비중보다 낮습니다."],
            "hard_errors": [],
            "orders": [],
        },
    }
    store.set_control("rebalance_latest", package)
    monkeypatch.setattr(
        "trading.controller.RebalanceExecutor.execute",
        lambda self, saved: {"status": "ORDERS_SUBMITTED", "orders": []},
    )

    try:
        controller.execute_rebalance(package["proposal_id"])
    except TradingControlError:
        pass
    else:
        raise AssertionError("Override 없는 거부 제안은 실행되면 안 됩니다.")

    result = controller.execute_rebalance(
        package["proposal_id"], override_risk=True
    )

    assert result["status"] == "ORDERS_RESERVED"
    events = store.list_audit_events(limit=10)
    assert any(item["event_type"] == "REBALANCE_RISK_OVERRIDE" for item in events)


def test_rebalance_preflight_blocks_all_orders_when_kill_switch_is_halted(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(service.context.config, rebalance_enabled=True)
    broker.buy("005930.KS", 100, 10)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    controller._ensure_manual_session()
    store.set_session_controls(
        controller.current_session_id(), kill_switch="HALTED", buy_enabled=True
    )
    package = {
        "proposal_id": "RB-TEST-HALTED",
        "created_at": controller.now().isoformat(),
        "validation": {
            "approved": True, "override_allowed": False,
            "errors": [], "hard_errors": [],
            "orders": [{
                "side": "SELL", "ticker": "005930.KS", "quantity": 10,
                "target_quantity": 0, "price": 100,
            }],
        },
    }
    store.set_control("rebalance_latest", package)

    try:
        controller.execute_rebalance(package["proposal_id"])
    except TradingControlError as error:
        assert "16번 메뉴" in str(error)
    else:
        raise AssertionError("HALTED 상태에서는 리밸런싱이 시작되면 안 됩니다.")

    assert broker.get_position("005930.KS") is not None
    assert store.get_control("rebalance_execution:RB-TEST-HALTED") is None


def test_buy_only_rebalance_ignores_unrelated_pending_sell(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(service.context.config, rebalance_enabled=True)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    controller._ensure_manual_session()
    store.create_order_intent(
        "old:SELL:000660.KS", controller.current_session_id(),
        "000660.KS", "SELL", {"quantity": 1},
    )
    store.update_order_intent(
        "old:SELL:000660.KS", "SUBMITTED", broker_order_id="OLD-ORDER"
    )
    package = {
        "snapshot": {"positions": []},
        "validation": {"orders": [{
            "side": "BUY", "ticker": "005930.KS", "name": "삼성전자",
            "quantity": 10, "original_quantity": 0, "target_quantity": 10,
            "price": 100, "sector": "반도체",
        }]},
    }

    result = RebalanceExecutor(controller).execute(package)

    assert result["status"] == "ORDERS_SUBMITTED"
    assert result["orders"][0]["status"] == "FILLED"
    assert broker.get_position("005930.KS").quantity == 10


def test_rebalance_reports_same_ticker_pending_sell_as_blocker(tmp_path):
    service, broker, store = _service(tmp_path)
    broker.buy("005930.KS", 100, 10)
    controller = TradingController(service)
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)
    store.create_order_intent(
        "old:SELL:005930.KS", controller.current_session_id(),
        "005930.KS", "SELL", {"quantity": 2},
    )
    store.update_order_intent(
        "old:SELL:005930.KS", "SUBMITTED", broker_order_id="OLD-SELL"
    )
    package = {
        "snapshot": {"positions": [{"ticker": "005930.KS", "quantity": 10}]},
        "validation": {"orders": [{
            "side": "SELL", "ticker": "005930.KS", "name": "삼성전자",
            "quantity": 1, "original_quantity": 10, "target_quantity": 9,
            "price": 100, "sector": "반도체",
        }]},
    }

    result = RebalanceExecutor(controller).execute(package)

    assert result["status"] == "AWAITING_SELL_FILLS"
    assert result["orders"] == []
    assert result["blocking_orders"][0]["broker_order_id"] == "OLD-SELL"
    assert result["blocking_orders"][0]["quantity"] == 2
    assert broker.get_position("005930.KS").quantity == 10


def test_approved_rebalance_sells_only_after_explicit_execution(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(
        service.context.config,
        rebalance_enabled=True,
        rebalance_max_turnover_pct=1.0,
    )
    broker.buy("005930.KS", 100, 10, sector="반도체")

    class Advisor:
        def propose(self, snapshot):
            return RebalanceProposal(
                market_view="RISK_OFF", market_summary="위험 회피",
                recommended_cash_pct=100, overall_reason="현금 확대",
                actions=[RebalanceAction(
                    ticker="005930.KS", action="SELL", target_weight_pct=0,
                    confidence=0.9, reason="위험 축소",
                )],
            )

    class Recommendations:
        calls = 0

        @staticmethod
        def top_recommendations(*args, **kwargs):
            Recommendations.calls += 1
            return [{
                "ticker": "005930.KS", "name": "삼성전자",
                "sector": "반도체", "rank": 1, "total_score": 80,
            }]

    class News:
        @staticmethod
        def collect():
            return {
                "sentiment": "NEGATIVE", "positive_hits": 0,
                "negative_hits": 1,
                "headlines": [{"title": "시장 위험 확대", "link": "test"}],
                "errors": [],
            }

        @staticmethod
        def collect_securities(securities):
            return {
                item["ticker"]: {
                    "ticker": item["ticker"], "name": item["name"],
                    "sentiment": "POSITIVE", "summary": "실적 전망 개선",
                    "headlines": [{
                        "title": "실적 개선 뉴스", "description": "전망 상향",
                        "link": "https://example.com/news",
                    }],
                }
                for item in securities
            }

    controller = TradingController(
        service,
        recommendation_service=Recommendations(),
        rebalance_advisor=Advisor(),
        market_news_service=News(),
    )
    controller.now = lambda: datetime(2025, 1, 6, 10, 0, tzinfo=SEOUL)

    controller.top_recommendations(universe_scope="KOSPI")
    package = controller.rebalance_proposal()

    assert Recommendations.calls == 1
    assert package["snapshot"]["recommendation_universe_scope"] == "KOSPI"
    assert package["validation"]["approved"] is True
    assert package["snapshot"]["security_news"]["005930.KS"]["sentiment"] == "POSITIVE"
    report = Path(package["report_path"])
    assert report.exists()
    assert report.name == "Rebalancing_Proposal_20250106.pdf"
    assert report.read_bytes().startswith(b"%PDF-")
    assert broker.get_position("005930.KS") is not None

    try:
        controller.execute_rebalance(package["proposal_id"])
    except TradingControlError as error:
        assert "종목별 주문 승인" in str(error)
    else:
        raise AssertionError("종목별 검토 전에는 리밸런싱이 실행되면 안 됩니다.")

    reviewed = controller.review_rebalance_orders(
        package["proposal_id"], package["validation"]["orders"]
    )
    assert reviewed["validation"]["individually_reviewed"] is True

    result = controller.execute_rebalance(package["proposal_id"])

    assert result["status"] == "ORDERS_RESERVED"
    assert broker.get_position("005930.KS") is not None
    queued = store.list_scheduled_orders(statuses=("QUEUED",))
    assert queued[0]["payload"]["order_type"] == "PRIORITY_LIMIT"
    service.run_scheduled_orders(datetime(2025, 1, 6, 10, 1, tzinfo=SEOUL))
    assert broker.get_position("005930.KS") is None
    assert store.get_control(f"rebalance_execution:{package['proposal_id']}")["status"] == "ORDERS_RESERVED"


def test_rebalance_reuses_saved_same_day_proposal_without_calling_llm(tmp_path):
    service, _, store = _service(tmp_path)
    service.context.config = replace(service.context.config, rebalance_enabled=True)
    saved = {
        "proposal_id": "RB-SAVED", "created_at": "2025-01-06T09:30:00+09:00",
        "proposal": {}, "validation": {"orders": []}, "snapshot": {},
    }
    store.set_control("rebalance_latest", saved)

    class Advisor:
        def propose(self, snapshot):
            raise AssertionError("저장 제안이 있으면 LLM을 다시 호출하면 안 됩니다")

    controller = TradingController(service, rebalance_advisor=Advisor())
    controller.now = lambda: datetime(2025, 1, 6, 14, 0, tzinfo=SEOUL)

    assert controller.rebalance_proposal() == saved


def test_user_feedback_revises_and_persists_rebalance_proposal(tmp_path):
    service, _, store = _service(tmp_path)
    service.context.config = replace(service.context.config, rebalance_enabled=True)
    snapshot = {
        "portfolio": {"total_equity": 1_000_000, "cash": 1_000_000},
        "positions": [], "top10": [{
            "ticker": "005930.KS", "name": "삼성전자", "sector": "반도체",
            "current_price": 100, "rank": 1, "total_score": 90,
            "recommendation_reason": "우수",
        }],
        "market_news": {"headlines": []},
    }
    original = RebalanceProposal(
        market_view="NEUTRAL", market_summary="기존", recommended_cash_pct=100,
        actions=[], overall_reason="기존 제안",
    )
    package = {
        "proposal_id": "RB-ORIGINAL", "created_at": "2025-01-06T10:00:00+09:00",
        "snapshot": snapshot, "proposal": original.model_dump(),
        "validation": RebalanceValidator(service.context.config).validate(snapshot, original),
        "requires_individual_review": True,
    }
    store.set_control("rebalance_latest", package)

    class Advisor:
        def revise(self, received_snapshot, current, feedback):
            assert received_snapshot == snapshot
            assert feedback == "삼성전자 비중을 20%로 조정"
            return RebalanceProposal(
                market_view="RISK_ON", market_summary="수정", recommended_cash_pct=80,
                actions=[RebalanceAction(
                    ticker="005930.KS", action="BUY", target_weight_pct=20,
                    confidence=0.9, reason="사용자 의견 반영",
                )], overall_reason="수정 제안",
            )

    controller = TradingController(service, rebalance_advisor=Advisor())
    controller.now = lambda: datetime(2025, 1, 6, 10, 5, tzinfo=SEOUL)

    revised = controller.revise_rebalance_proposal(
        "RB-ORIGINAL", "삼성전자 비중을 20%로 조정"
    )

    assert revised["proposal_id"] != "RB-ORIGINAL"
    assert revised["revision"] == 1
    assert revised["proposal"]["recommended_cash_pct"] == 80
    assert revised["validation"]["orders"][0]["ticker"] == "005930.KS"
    assert revised["validation"].get("individually_reviewed") is not True
    assert revised["revision_history"][0]["feedback"] == "삼성전자 비중을 20%로 조정"
    assert store.get_control("rebalance_latest")["proposal_id"] == revised["proposal_id"]
    assert Path(revised["report_path"]).read_bytes().startswith(b"%PDF-")


def test_after_close_rebalance_is_reserved_and_executes_next_trading_day(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(
        service.context.config, rebalance_enabled=True, dry_run=False
    )
    service.config = service.context.config
    broker.buy("005930.KS", 100, 5)
    broker.set_market_price("000660.KS", 200)
    controller = TradingController(service)
    friday_close = datetime(2025, 1, 10, 16, 0, tzinfo=SEOUL)
    controller.now = lambda: friday_close
    package = {
        "proposal_id": "RB-AFTER-CLOSE",
        "created_at": friday_close.isoformat(),
        "requires_individual_review": True,
        "validation": {
            "approved": True,
            "individually_reviewed": True,
            "orders": [
                {
                    "ticker": "005930.KS", "name": "삼성전자",
                    "side": "SELL", "quantity": 2, "sector": "반도체",
                },
                {
                    "ticker": "000660.KS", "name": "SK하이닉스",
                    "side": "BUY", "quantity": 3, "sector": "반도체",
                },
            ],
        },
    }
    store.set_control("rebalance_latest", package)

    reserved = controller.execute_rebalance("RB-AFTER-CLOSE")

    assert reserved["status"] == "ORDERS_RESERVED"
    assert reserved["execute_on"] == "2025-01-13"
    queue = controller.scheduled_orders()
    assert [item["side"] for item in queue] == ["SELL", "BUY"]
    assert broker.get_position("005930.KS").quantity == 5
    assert broker.get_position("000660.KS") is None

    monday_open = datetime(2025, 1, 13, 9, 0, tzinfo=SEOUL)
    result = service.run_scheduled_orders(monday_open)

    assert result.status == "COMPLETED"
    assert [item["side"] for item in result.result["orders"]] == ["SELL", "BUY"]
    assert broker.get_position("005930.KS").quantity == 3
    assert broker.get_position("000660.KS").quantity == 3
    assert controller.scheduled_orders() == []
    completed = store.list_scheduled_orders(statuses=("SUBMITTED",))
    assert len(completed) == 2


def test_scheduled_order_state_survives_trade_csv_failure(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(service.context.config, dry_run=False)
    service.config = service.context.config
    broker.set_market_price("000660.KS", 200)

    class FailingTradeLogger:
        def log(self, result):
            raise PermissionError("trades.csv is locked")

    service.context.trade_logger = FailingTradeLogger()
    store.enqueue_scheduled_order(
        "RB-LOG:BUY:000660.KS", "RB-LOG", "2025-01-13",
        "000660.KS", "BUY", 3, {"sector": "semiconductor"},
    )

    result = service.run_scheduled_orders(
        datetime(2025, 1, 13, 9, 0, tzinfo=SEOUL)
    )

    assert result.status == "COMPLETED"
    submitted = store.list_scheduled_orders(statuses=("SUBMITTED",))
    assert len(submitted) == 1
    intents = store.list_order_intents(limit=10)
    assert len(intents) == 1
    assert intents[0]["status"] != "INTENT"
    assert submitted[0]["payload"]["status"] == intents[0]["status"]
    assert broker.get_position("000660.KS").quantity == 3


def test_scheduled_orders_can_be_amended_or_cancelled_before_open(tmp_path):
    service, broker, store = _service(tmp_path)
    service.context.config = replace(service.context.config, dry_run=False)
    service.config = service.context.config
    broker.buy("005930.KS", 100, 5)
    broker.set_market_price("000660.KS", 200)
    controller = TradingController(service)
    execute_on = "2025-01-13"
    store.enqueue_scheduled_order(
        "RB-EDIT:SELL:005930.KS", "RB-EDIT", execute_on,
        "005930.KS", "SELL", 2, {"name": "삼성전자", "sector": "반도체"},
    )
    store.enqueue_scheduled_order(
        "RB-EDIT:BUY:000660.KS", "RB-EDIT", execute_on,
        "000660.KS", "BUY", 3, {"name": "SK하이닉스", "sector": "반도체"},
    )

    amended = controller.amend_scheduled_order(
        "RB-EDIT:SELL:005930.KS", 4
    )
    cancelled = controller.cancel_scheduled_order(
        "RB-EDIT:BUY:000660.KS"
    )

    assert amended["quantity"] == 4
    assert cancelled["status"] == "CANCELLED"
    assert [item["quantity"] for item in controller.scheduled_orders()] == [4]

    result = service.run_scheduled_orders(
        datetime(2025, 1, 13, 9, 0, tzinfo=SEOUL)
    )

    assert [item["side"] for item in result.result["orders"]] == ["SELL"]
    assert result.result["orders"][0]["quantity"] == 4
    assert broker.get_position("005930.KS").quantity == 1
    assert broker.get_position("000660.KS") is None
    assert store.list_scheduled_orders(statuses=("CANCELLED",))[0][
        "reservation_id"
    ] == "RB-EDIT:BUY:000660.KS"


def test_reviewed_rebalance_can_change_quantity_and_exclude_orders(tmp_path):
    config = replace(
        LiveTradingConfig(), rebalance_max_turnover_pct=1.0,
        rebalance_max_position_pct=1.0, rebalance_max_sector_pct=1.0,
        rebalance_min_cash_pct=0.0,
    )
    snapshot = {
        "portfolio": {"total_equity": 1_000_000, "cash": 500_000},
        "positions": [{
            "ticker": "005930.KS", "name": "삼성전자", "quantity": 10,
            "current_price": 50_000, "market_value": 500_000,
            "weight_pct": 50, "sector": "반도체",
        }],
        "top10": [{
            "ticker": "000660.KS", "name": "SK하이닉스",
            "current_price": 100_000, "sector": "반도체",
        }],
    }
    proposal = RebalanceProposal(
        market_view="NEUTRAL", market_summary="중립", recommended_cash_pct=0,
        overall_reason="교체", actions=[],
    )

    validation = RebalanceValidator(config).validate_reviewed_orders(
        snapshot, proposal, [{
            "ticker": "005930.KS", "side": "SELL", "quantity": 3,
            "confidence": 0.8, "reason": "사용자 축소",
        }]
    )

    assert validation["approved"] is True
    assert validation["individually_reviewed"] is True
    assert validation["orders"][0]["quantity"] == 3
    assert validation["orders"][0]["target_quantity"] == 7
    assert validation["orders"][0]["estimated_value"] == 150_000


def test_console_rebalance_review_supports_approve_modify_and_exclude(
    tmp_path, monkeypatch
):
    service, _, _ = _service(tmp_path)
    console = TradingConsole(TradingController(service))
    orders = [
        {
            "ticker": "005930.KS", "name": "삼성전자", "side": "SELL",
            "quantity": 3, "original_quantity": 10, "price": 50_000,
            "estimated_value": 150_000, "reason": "비중 축소",
        },
        {
            "ticker": "000660.KS", "name": "SK하이닉스", "side": "BUY",
            "quantity": 2, "original_quantity": 0, "price": 100_000,
            "estimated_value": 200_000, "reason": "신규 편입",
        },
        {
            "ticker": "035420.KS", "name": "NAVER", "side": "BUY",
            "quantity": 1, "original_quantity": 0, "price": 200_000,
            "estimated_value": 200_000, "reason": "후보 편입",
        },
    ]
    answers = iter(["A", "M", "BUY", "4", "X"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    reviewed = console._review_rebalance_orders(orders)

    assert reviewed is not None
    assert len(reviewed) == 2
    assert reviewed[0]["quantity"] == 3
    assert reviewed[1]["side"] == "BUY"
    assert reviewed[1]["quantity"] == 4
    assert "사용자 수정" in reviewed[1]["reason"]
