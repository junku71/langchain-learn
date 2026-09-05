from __future__ import annotations

from dataclasses import asdict

from trading.controller import TradingControlError, TradingController
from trading.display import korean_time, money, stock_name, table
from trading.factory import create_live_trading_service
from trading.market import get_market_profile


def _ticker(value: str) -> str:
    value = value.strip().upper()
    if not value:
        raise TradingControlError("종목코드를 입력하세요.")
    return f"{value}.KS" if "." not in value else value


def _number(prompt: str, *, default: float | None = None) -> float:
    suffix = f" [{default:g}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip().replace(",", "")
    if not raw and default is not None:
        return float(default)
    return float(raw)


def _optional_number(prompt: str, default: float | None = None) -> float | None:
    shown = "없음" if default is None else f"{default:g}"
    raw = input(f"{prompt} [{shown}, '-' 입력 시 해제]: ").strip().replace(",", "")
    if raw == "-":
        return None
    if not raw:
        return default
    return float(raw)


class TradingConsole:
    def __init__(self, controller: TradingController):
        self.controller = controller

    @property
    def currency_label(self) -> str:
        return "원" if self.controller.context.config.currency == "KRW" else " USD"

    def run(self) -> None:
        while True:
            try:
                self._header()
                self._menu()
                choice = input("선택 > ").strip()
                if choice == "0":
                    self.controller.stop_scheduler()
                    print("콘솔을 종료합니다.")
                    return
                self._dispatch(choice)
            except (TradingControlError, ValueError) as error:
                print(f"\n[입력/제어 오류] {error}")
            except KeyboardInterrupt:
                print("\n작업을 취소했습니다.")
            except Exception as error:
                print(f"\n[실행 오류] {type(error).__name__}: {error}")
            input("\nEnter를 누르면 메인 메뉴로 돌아갑니다...")

    def _header(self) -> None:
        env = self.controller.environment()
        profile = get_market_profile(self.controller.context.config.market_region)
        print("\n" + "=" * 110)
        print(f" {profile.display_name} 자동매매 제어 콘솔")
        print("=" * 110)
        print(
            f" Broker: {env['broker']} / {env['account_type']}    "
            f"Trading: {'ENABLED' if env['trading_enabled'] else 'DISABLED'}    "
            f"Dry Run: {env['dry_run']}"
        )
        print(
            f" Market: {env['market_phase']}    Scheduler: {env['scheduler']}    "
            f"Buy: {'ON' if env['buy_enabled'] else 'OFF'}    "
            f"ML Filter: {env['ml_filter']}    Slack: {env['slack']}    "
            f"Kill Switch: {env['kill_switch']}"
        )
        if env["account_type"] == "REAL" and not env["dry_run"]:
            print(" !!! REAL ACCOUNT: 실제 자금 주문 모드입니다 !!!")
        print("=" * 110)

    @staticmethod
    def _menu() -> None:
        print(
            """
 [주문]
    1. 예약종목 조회 및 취소/정정
    2. 사용자 지정 종목명 매수
    3. 사용자 지정 종목 매도    4. 미체결 주문 조회 및 취소
    5. 최근주문체결내역

 [자동매매]
    6. 매도전략 선택(샹들리에|돈치안|직접지정)
    7. 자동매매 작업 수동 실행

 [포트폴리오 관리]
    8. Reconciliation
    9. 잔고 및 보유종목 조회    10. 오늘의 Top10 추천 및 결과저장
    11. 보유 및 Top10 추천종목 리밸런싱 제안 및 승인 후 주문예약

 [설정관리]
    12. 시스템 실행 상태
    13. Scheduler 시작·중지 Toggle
    14. 감사 로그
    15. 전체 Kill Switch         16. ML Filter ON/OFF
    17. Slack 알림 연결 테스트

    0. 종료
"""
        )

    def _dispatch(self, choice: str) -> None:
        actions = {
            "1": self._candidates,
            "2": self._buy,
            "3": self._sell_menu,
            "4": self._cancel_pending,
            "5": self._orders,
            "6": self._protection_menu,
            "7": self._run_job,
            "8": self._api_reconciliation,
            "9": self._account_overview,
            "10": self._top_recommendations,
            "11": self._rebalance,
            "12": self._system_status,
            "13": self._scheduler_toggle,
            "14": self._audit,
            "15": self._toggle_kill,
            "16": self._toggle_ml_filter,
            "17": self._test_slack,
        }
        action = actions.get(choice)
        if action is None:
            raise TradingControlError("메뉴 번호를 다시 확인하세요.")
        action()

    def _account_overview(self) -> None:
        """Display the balance and position outputs consecutively."""
        self._balance()
        self._positions()

    def _balance(self) -> None:
        balance, positions = self.controller.account_snapshot()
        print("\n[잔고 요약]")
        print(table(
            [
                "총평가금액", "주식평가금액", "주식매입금액", "예수금",
                "D+2예수금", "실현손익", "평가손익",
            ],
            [[
                money(balance.get("total_equity")),
                money(balance.get("stock_market_value")),
                money(balance.get("stock_purchase_amount")),
                money(balance.get("cash")), money(balance.get("d2_cash")),
                money(balance.get("realized_pnl")),
                money(balance.get("unrealized_pnl")),
            ]],
        ))

    def _positions(self) -> None:
        _, positions = self.controller.account_snapshot()
        rows = []
        total_purchase = 0.0
        total_current = 0.0
        quote_failures = 0
        for ticker, position in positions.items():
            protection = self.controller.context.store.get_protection(ticker)
            purchase_amount = position.avg_price * position.quantity
            total_purchase += purchase_amount
            try:
                current_price = self.controller.quote(ticker)
                current_amount = current_price * position.quantity
                profit_amount = current_amount - purchase_amount
                return_pct = (
                    profit_amount / purchase_amount
                    if purchase_amount > 0 else None
                )
                total_current += current_amount
            except Exception:
                current_price = None
                current_amount = None
                profit_amount = None
                return_pct = None
                quote_failures += 1
            rows.append([
                ticker, stock_name(ticker), f"{position.quantity:,}", money(position.avg_price),
                money(purchase_amount), money(current_price), money(current_amount),
                f"{return_pct:.2%}" if return_pct is not None else "-",
                money(profit_amount),
                money(protection.stop_loss if protection else position.stop_loss),
                money(protection.take_profit if protection else position.take_profit),
                (
                    f"{(protection.trailing_stop_pct if protection else position.trailing_stop_pct):.1%}"
                    if (protection.trailing_stop_pct if protection else position.trailing_stop_pct) is not None
                    else "미설정"
                ),
            ])
        print("\n[보유종목 및 보호가격]")
        print(table(
            ["종목코드", "종목명", "수량", "평균단가", "종목별 총구매가", "현재가", "종목별 총현재가", "수익률", "손익금액", "손절가", "익절가", "Trailing"],
            rows,
        ))
        portfolio_return = (
            (total_current - total_purchase) / total_purchase
            if total_purchase > 0 and quote_failures == 0 else None
        )
        print("\n[포트폴리오 합계]")
        print(table(
            ["총구입금액", "현재 총주식금액", "총손익금액", "포트폴리오 총수익률"],
            [[
                money(total_purchase),
                money(total_current) if quote_failures == 0 else "-",
                money(total_current - total_purchase) if quote_failures == 0 else "-",
                f"{portfolio_return:.2%}" if portfolio_return is not None else "-",
            ]],
        ))
        if quote_failures:
            print(f"※ 현재가 조회 실패 {quote_failures}개 종목이 있어 현재 평가 합계 표시를 생략했습니다.")

    def _order_table(self, orders: list[dict]) -> None:
        def total_amount(item: dict) -> str:
            payload = item.get("payload", {})
            try:
                return money(float(payload["price"]) * int(payload["quantity"]))
            except (KeyError, TypeError, ValueError):
                return "-"

        print(table(
            [
                "시각", "종목코드", "종목명", "구분", "상태", "가격", "수량",
                "총주문금액", "주문번호",
            ],
            [[
                korean_time(item["updated_at"]), item["ticker"],
                stock_name(item["ticker"]), item["side"],
                item["status"], money(item.get("payload", {}).get("price")),
                item.get("payload", {}).get("quantity", "-"), total_amount(item),
                item.get("broker_order_id") or "-",
            ] for item in orders],
        ))

    def _pending(self) -> None:
        print("\n[Reconciliation 대상 주문]")
        self._order_table(self.controller.pending_orders())
        print("※ KIS 체결조회 API reconciliation 전까지는 실제 미체결 여부와 다를 수 있습니다.")

    def _orders(self) -> None:
        print("\n[KIS 주문체결 reconciliation]")
        try:
            report = self.controller.api_reconciliation()
            print(table(
                ["조회 주문", "상태 변경", "Slack 알림", "오류"],
                [[
                    report.get("checked", 0), report.get("changed", 0),
                    report.get("notified", 0), report.get("errors", 0),
                ]],
            ))
        except TradingControlError as error:
            # PaperBroker 또는 일시적인 KIS 장애에서도 로컬 주문 이력은 확인할 수 있다.
            print(f"※ reconciliation을 완료하지 못했습니다: {error}")
        print("\n[최근 주문 내역]")
        self._order_table(self.controller.order_history())

    def _api_reconciliation(self) -> None:
        print("\n[KIS API reconciliation 실행]")
        report = self.controller.api_reconciliation()
        print(table(
            ["조회 주문", "상태 변경", "Slack 알림", "오류"],
            [[
                report.get("checked", 0),
                report.get("changed", 0),
                report.get("notified", 0),
                report.get("errors", 0),
            ]],
        ))
        print("\n[reconciliation 후 미체결 주문]")
        pending = self.controller.pending_orders()
        if pending:
            self._order_table(pending)
        else:
            print("미체결 주문이 없습니다.")

    def _system_status(self) -> None:
        env = self.controller.environment()
        session = self.controller.current_session() or {}
        print("\n[시스템 상태]")
        print(table(["항목", "값"], [[key, value] for key, value in env.items()]))
        print(f"Session: {session.get('session_id', '없음')}")

    def _candidates(self) -> None:
        reservations = self.controller.scheduled_orders()
        print("\n[Buy/Sell 예약주문 Queue]")
        if reservations:
            print(table(
                ["번호", "실행일", "구분", "종목코드", "종목명", "수량", "상태", "제안서"],
                [[
                    index, item.get("execute_on"), item.get("side"), item.get("ticker"),
                    item.get("payload", {}).get("name") or stock_name(item.get("ticker", "")),
                    f"{int(item.get('quantity', 0)):,}", item.get("status"),
                    item.get("proposal_id"),
                ] for index, item in enumerate(reservations, 1)],
            ))
            selected = input(
                "취소/정정할 예약 번호 (조회만 하려면 Enter): "
            ).strip()
            if selected:
                try:
                    selected_index = int(selected)
                    if not 1 <= selected_index <= len(reservations):
                        raise ValueError
                    item = reservations[selected_index - 1]
                except ValueError:
                    raise TradingControlError("예약 주문 번호를 다시 확인하세요.")
                action = input("C(취소), M(수량 정정), Enter(돌아가기): ").strip().upper()
                if action == "C":
                    if input("예약을 취소하려면 CANCEL 입력: ").strip().upper() == "CANCEL":
                        self.controller.cancel_scheduled_order(item["reservation_id"])
                        print("예약 주문을 취소했습니다.")
                    else:
                        print("예약 취소를 중단했습니다.")
                elif action == "M":
                    quantity = int(_number("변경할 수량"))
                    if input("수량을 정정하려면 MODIFY 입력: ").strip().upper() == "MODIFY":
                        self.controller.amend_scheduled_order(
                            item["reservation_id"], quantity
                        )
                        print(f"예약 주문 수량을 {quantity:,}주로 정정했습니다.")
                    else:
                        print("예약 정정을 중단했습니다.")
                elif action:
                    raise TradingControlError("C, M 또는 Enter 중 하나를 입력하세요.")
        else:
            print("예약된 Buy/Sell 주문이 없습니다.")
        candidates = self.controller.candidates()
        print("\n[장전 BUY 후보종목]")
        print(table(
            ["종목코드", "종목명", "시장", "섹터", "ML 점수", "확률", "순위"],
            [[
                item.get("ticker"), item.get("name") or stock_name(item.get("ticker", "")),
                item.get("market"), item.get("sector"),
                f"{float(item.get('ml_score', 0)):.4f}",
                f"{float(item.get('classification_probability', 0)):.1%}",
                item.get("ml_rank"),
            ] for item in candidates],
        ))

    def _confirm_order(self, side: str, ticker: str) -> bool:
        env = self.controller.environment()
        required = f"{side} {ticker.split('.')[0]}"
        if env["account_type"] == "REAL" and not env["dry_run"]:
            print("실계좌 주문입니다. 취소하려면 아무 값이나 입력하세요.")
            return input(f"확인 문구 `{required}` 입력: ").strip().upper() == required
        return input("주문을 제출하려면 YES 입력: ").strip().upper() == "YES"

    def _buy(self) -> None:
        requested_name = input("매수할 종목명: ").strip()
        if not requested_name:
            raise TradingControlError("종목명을 입력하세요.")
        security = self.controller.resolve_security(requested_name)
        ticker = security["ticker"]
        current = self.controller.quote(ticker)
        name = security["name"]
        sector = security["sector"]
        print(f"종목: {ticker} {name} / 현재가: {money(current)}{self.currency_label}")
        quantity = int(_number("매수 수량"))
        order_type, price = self._buy_price_method(current)
        stop = _optional_number("손절가", round(price * 0.95))
        target = _optional_number("익절가", round(price * 1.10))
        trailing_pct = _optional_number(
            "Trailing stop(%)", self.controller.context.config.trailing_stop_pct * 100
        )
        trailing = trailing_pct / 100 if trailing_pct is not None else None
        print("\n[매수 주문 확인]")
        print(table(["종목코드", "종목명", "가격방식", "기준가격", "수량", "주문금액", "손절가", "익절가", "Trailing"], [[
            ticker, name, order_type, money(price), quantity, money(price * quantity), money(stop),
            money(target), f"{trailing:.1%}" if trailing is not None else "없음",
        ]]))
        if not self._confirm_order("BUY", ticker):
            print("주문을 취소했습니다.")
            return
        result = self.controller.manual_buy_or_reserve(
            ticker, quantity, limit_price=price, sector=sector, name=name,
            stop_loss=stop, take_profit=target, trailing_stop_pct=trailing,
            order_type=order_type,
        )
        if isinstance(result, dict):
            print(
                f"예약 결과: {result['status']} / 실행일: {result['execute_on']} / "
                f"예약번호: {result['reservation_id']}"
            )
        else:
            print(f"주문 결과: {result.status} / 주문번호: {result.order_id or '-'} / {result.reason}")

    def _select_position(self) -> tuple[str, object]:
        positions = self.controller.context.broker.get_positions()
        items = list(positions.items())
        print(table(["번호", "종목코드", "종목명", "수량", "평균단가", "총평가금액"], [
            [
                index, ticker, stock_name(ticker), position.quantity,
                money(position.avg_price), money(position.avg_price * position.quantity),
            ]
            for index, (ticker, position) in enumerate(items, 1)
        ]))
        if not items:
            raise TradingControlError("보유종목이 없습니다.")
        index = int(_number("종목 번호"))
        if not 1 <= index <= len(items):
            raise TradingControlError("종목 번호가 올바르지 않습니다.")
        return items[index - 1]

    @staticmethod
    def _buy_price_method(current: float) -> tuple[str, float]:
        print("\n[매수가격 방식]")
        print("1. 시장가")
        print("2. 지정가")
        print("3. 최우선지정가 (가격우선)")
        print("4. 최유리지정가 (타이밍우선)")
        order_type = {
            "1": "MARKET", "2": "LIMIT",
            "3": "PRIORITY_LIMIT", "4": "BEST_LIMIT",
        }.get(input("선택 > ").strip())
        if order_type is None:
            raise TradingControlError("매수가격 방식 번호를 다시 확인하세요.")
        price = _number("지정가", default=current) if order_type == "LIMIT" else current
        return order_type, price

    @staticmethod
    def _sell_price_method(current: float) -> tuple[str, float]:
        print("\n[매도가격 방식]")
        print("1. 시장가")
        print("2. 지정가")
        print("3. 최우선지정가 (가격우선)")
        print("4. 최유리지정가 (타이밍우선)")
        order_type = {
            "1": "MARKET", "2": "LIMIT",
            "3": "PRIORITY_LIMIT", "4": "BEST_LIMIT",
        }.get(input("선택 > ").strip())
        if order_type is None:
            raise TradingControlError("매도가격 방식 번호를 다시 확인하세요.")
        price = _number("지정가", default=current) if order_type == "LIMIT" else current
        return order_type, price

    def _sell(self) -> None:
        ticker, position = self._select_position()
        current = self.controller.quote(ticker)
        quantity = int(_number("매도 수량", default=position.quantity))
        order_type, price = self._sell_price_method(current)
        print(
            f"{ticker} {stock_name(ticker)} {quantity:,}주 / "
            f"{order_type} / 기준가격 {money(price)}{self.currency_label}"
        )
        if not self._confirm_order("SELL", ticker):
            print("주문을 취소했습니다.")
            return
        result = self.controller.manual_sell(
            ticker, quantity, limit_price=price, order_type=order_type
        )
        print(f"주문 결과: {result.status} / 주문번호: {result.order_id or '-'} / {result.reason}")

    def _sell_menu(self) -> None:
        print("\n[사용자 지정 종목 매도]")
        print("1. 수량 지정 매도")
        print("2. 선택 종목 전량 매도")
        choice = input("선택 > ").strip()
        if choice == "1":
            self._sell()
        elif choice == "2":
            self._sell_all()
        else:
            raise TradingControlError("매도 서브메뉴 번호를 다시 확인하세요.")

    def _sell_all(self) -> None:
        ticker, position = self._select_position()
        current = self.controller.quote(ticker)
        order_type, price = self._sell_price_method(current)
        print(
            f"{ticker} {stock_name(ticker)} 전량 {position.quantity:,}주를 "
            f"{order_type} 방식으로 매도합니다."
        )
        if not self._confirm_order("SELL", ticker):
            print("주문을 취소했습니다.")
            return
        result = self.controller.manual_sell(
            ticker, position.quantity, limit_price=price, order_type=order_type
        )
        print(f"주문 결과: {result.status} / 주문번호: {result.order_id or '-'}")

    @staticmethod
    def _cancel_info() -> None:
        print("KIS 주문 정정·취소 및 실제 미체결 조회 API는 아직 Broker에 구현되지 않았습니다.")
        print("잘못된 취소 성공 표시를 방지하기 위해 이 메뉴에서는 주문을 전송하지 않습니다.")

    def _cancel_pending(self) -> None:
        orders = self.controller.pending_orders()
        print("\n[미체결 주문 조회 및 취소]")
        if not orders:
            print("취소할 수 있는 미체결 주문이 없습니다.")
            return
        print(table(
            ["번호", "시각", "종목코드", "종목명", "구분", "상태", "잔량", "가격", "KIS 주문번호"],
            [[
                index,
                korean_time(item["updated_at"]),
                item["ticker"],
                stock_name(item["ticker"]),
                item["side"],
                item["status"],
                item.get("payload", {}).get("remaining_quantity")
                or item.get("payload", {}).get("quantity", "-"),
                money(item.get("payload", {}).get("price")),
                item.get("broker_order_id") or "-",
            ] for index, item in enumerate(orders, start=1)],
        ))
        selection = input("취소할 주문 번호 (조회만 하려면 Enter): ").strip()
        if not selection:
            return
        try:
            index = int(selection)
        except ValueError as exc:
            raise TradingControlError("주문 번호는 숫자로 입력하세요.") from exc
        if not 1 <= index <= len(orders):
            raise TradingControlError("주문 번호가 올바르지 않습니다.")
        intent = orders[index - 1]
        broker_order_id = intent.get("broker_order_id") or ""
        required = f"CANCEL {broker_order_id}"
        print("선택한 원 주문의 미체결 잔량 전체를 취소 요청합니다.")
        confirmed = input(f"확인 문구 `{required}` 입력: ").strip().upper()
        if confirmed != required.upper():
            print("취소 요청을 철회했습니다.")
            return
        outcome = self.controller.cancel_pending_order(intent["idempotency_key"])
        request = outcome["request"]
        order = outcome["order"]
        print(
            f"취소 요청 결과: {request['status']} / "
            f"취소 주문번호: {request.get('order_id') or '-'} / "
            f"{request.get('reason') or '-'}"
        )
        print(f"원 주문 현재 상태: {order.get('status', '-')}")
        if request["status"] == "CANCEL_SUBMITTED" and order.get("status") != "CANCELLED":
            print("KIS 취소 접수 후 reconciliation에서 최종 CANCELLED 상태를 확인합니다.")

    def _protection(self) -> None:
        ticker, position = self._select_position()
        current = self.controller.quote(ticker)
        return_pct = (
            (current - position.avg_price) / position.avg_price
            if position.avg_price > 0 else None
        )
        return_text = f"{return_pct:.2%}" if return_pct is not None else "-"
        print(
            f"선택 종목: {ticker} {stock_name(ticker)} / "
            f"평균단가 {money(position.avg_price)}{self.currency_label} / "
            f"현재가 {money(current)}{self.currency_label} / "
            f"손익률 {return_text}"
        )
        previous = self.controller.context.store.get_protection(ticker)
        stop = _optional_number("새 손절가", previous.stop_loss if previous else position.stop_loss)
        target = _optional_number("새 익절가", previous.take_profit if previous else position.take_profit)
        old_trailing = previous.trailing_stop_pct if previous else position.trailing_stop_pct
        trailing_pct = _optional_number(
            "새 Trailing stop(%)", old_trailing * 100 if old_trailing is not None else None
        )
        state = self.controller.set_protection(
            ticker, stop_loss=stop, take_profit=target,
            trailing_stop_pct=trailing_pct / 100 if trailing_pct is not None else None,
        )
        print(f"보호가격을 저장했습니다: {asdict(state)}")

    def _protection_menu(self) -> None:
        print("\n[현재 설정된 매도전략]")
        overview = self.controller.exit_strategy_overview()
        print(table(
            ["번호", "종목명", "수량", "평균단가", "현재가격", "수익률", "총평가금액", "매도전략명", "ATR", "익절가", "손절가", "Trailing Stop가격", "상태"],
            [[
                index, item["name"], f"{item['quantity']:,}", money(item["avg_price"]),
                money(item["current_price"]),
                f"{item['return_pct']:.2%}" if item["return_pct"] is not None else "-",
                money(item["total_value"]), item["strategy_name"], money(item["atr"]),
                money(item["take_profit"]), money(item["stop_loss"]),
                money(item["trailing_stop"]), item["message"] or item["status"],
            ] for index, item in enumerate(overview, 1)],
        ))
        print("\n1. 종목별 매도전략 선택")
        print("2. 매도전략 일괄 선택")
        choice = input("선택 > ").strip()
        if choice == "1":
            raw = input("전략을 변경할 종목 번호(복수 선택은 쉼표 구분): ").strip()
            try:
                indexes = []
                for token in raw.split(","):
                    index = int(token.strip())
                    if not 1 <= index <= len(overview):
                        raise ValueError
                    if index not in indexes:
                        indexes.append(index)
            except ValueError as error:
                raise TradingControlError(
                    "종목 번호를 쉼표로 구분해 입력하세요. 예: 1,3,5"
                ) from error
            if not indexes:
                raise TradingControlError("선택한 종목이 없습니다.")
            tickers = [overview[index - 1]["ticker"] for index in indexes]
            self._select_bulk_exit_strategy(tickers=tickers, selected_only=True)
        elif choice == "2":
            self._select_bulk_exit_strategy()
        else:
            raise TradingControlError("매도전략 메뉴 번호를 다시 확인하세요.")

    def _exit_strategy_parameters(
        self,
    ) -> tuple[str, float, int, float, float, float]:
        print("1. 샹들리에 Exit (3ATR 초기손절 → 최고가-3ATR 추적 → 고정익절 없음)")
        print("2. 돈치안 추세추종 (20일 저가 채널 이탈 → 고정익절 없음)")
        print("3. 직접지정전략 (익절 +20% / 손절 -10% / Trailing Stop 8%)")
        choice = input("전략 선택 > ").strip()
        if choice == "1":
            return (
                "CHANDELIER_EXIT", _number("ATR 배수", default=3.0),
                20, 0.20, 0.10, 0.08,
            )
        if choice == "2":
            return (
                "DONCHIAN_TREND", 3.0,
                int(_number("돈치안 기간(거래일)", default=20)),
                0.20, 0.10, 0.08,
            )
        if choice == "3":
            take_profit_pct = _number("익절률(%)", default=20.0) / 100
            stop_loss_pct = _number("손절률(%)", default=10.0) / 100
            trailing_stop_pct = _number("Trailing Stop(%)", default=8.0) / 100
            return (
                "DIRECT_SPECIFIED", 3.0, 20,
                take_profit_pct, stop_loss_pct, trailing_stop_pct,
            )
        raise TradingControlError("매도전략 번호를 다시 확인하세요.")

    def _select_exit_strategy(self, ticker: str) -> None:
        previous = self.controller.context.store.get_protection(ticker)
        strategy_names = {
            "CHANDELIER_EXIT": "샹들리에 Exit",
            "DONCHIAN_TREND": "돈치안 추세추종",
            "DIRECT_SPECIFIED": "직접지정전략",
            "LEGACY": "기존 보호가격",
        }
        current_strategy = strategy_names.get(
            previous.strategy if previous else "LEGACY", "미설정"
        )
        print(f"현재 선택 전략: {current_strategy}")
        (
            strategy, atr_multiple, donchian_period,
            take_profit_pct, stop_loss_pct, trailing_stop_pct,
        ) = self._exit_strategy_parameters()
        print("ATR 및 보호가격을 계산하고 있습니다...")
        preview = self.controller.preview_exit_strategy(
            ticker, strategy, atr_multiple=atr_multiple,
            donchian_period=donchian_period,
            direct_take_profit_pct=take_profit_pct,
            direct_stop_loss_pct=stop_loss_pct,
            direct_trailing_stop_pct=trailing_stop_pct,
        )
        print(table(
            ["종목명", "선택 전략", "ATR", "손절가", "익절가", "Trailing stop", "상태"],
            [[
                preview["name"], preview["strategy_name"], money(preview["atr"]),
                money(preview["stop_loss"]), money(preview["take_profit"]),
                money(preview["trailing_stop"]), preview["warning"] or "적용 가능",
            ]],
        ))
        if input("이 전략을 적용하려면 APPLY 입력: ").strip().upper() != "APPLY":
            print("전략 적용을 취소했습니다. 변경된 값은 없습니다.")
            return
        state = self.controller.apply_exit_strategy(preview)
        print(table(
            ["종목명", "선택 전략", "ATR", "손절가", "익절가", "Trailing stop"],
            [[
                stock_name(ticker), preview["strategy_name"], money(state.atr),
                money(state.stop_loss), money(state.take_profit),
                money(state.trailing_stop),
            ]],
        ))

    def _select_bulk_exit_strategy(
        self, *, tickers: list[str] | None = None, selected_only: bool = False
    ) -> None:
        (
            strategy, atr_multiple, donchian_period,
            take_profit_pct, stop_loss_pct, trailing_stop_pct,
        ) = self._exit_strategy_parameters()
        print("전체 보유종목의 ATR 및 보호가격을 계산하고 있습니다...")
        previews = self.controller.preview_bulk_exit_strategy(
            strategy, tickers=tickers, atr_multiple=atr_multiple,
            donchian_period=donchian_period,
            direct_take_profit_pct=take_profit_pct,
            direct_stop_loss_pct=stop_loss_pct,
            direct_trailing_stop_pct=trailing_stop_pct,
        )
        print(table(
            ["종목명", "선택 전략", "ATR", "익절가", "손절가", "Trailing Stop가격", "상태"],
            [[
                item["name"], item["strategy_name"], money(item["atr"]),
                money(item["take_profit"]), money(item["stop_loss"]),
                money(item["trailing_stop"]), item["message"] or item["status"],
            ] for item in previews],
        ))
        confirmation = "APPLY SELECTED" if selected_only else "APPLY ALL"
        if input(f"적용하려면 {confirmation} 입력: ").strip().upper() != confirmation:
            print("일괄 적용을 취소했습니다. 변경된 값은 없습니다.")
            return
        results = self.controller.apply_bulk_exit_strategy(previews)
        saved = sum(item["status"] == "SAVED" for item in results)
        failed = sum(item["status"] == "ERROR" for item in results)
        scope = "선택 종목" if selected_only else "전체 종목"
        print(f"매도전략 적용 완료({scope}): 저장 {saved}개 / 실패 {failed}개")

    def _bulk_protection(self) -> None:
        positions = self.controller.context.broker.get_positions()
        if not positions:
            raise TradingControlError("보유종목이 없습니다.")
        print("\n다음 공식으로 모든 보유종목의 보호가격을 다시 계산합니다.")
        print("손절가 = 평균단가 - 3 × 일봉 ATR")
        print("익절가 = 평균단가 × 1.20")
        trailing_pct = _number(
            "Trailing stop(%)",
            default=self.controller.context.config.trailing_stop_pct * 100,
        )
        if not 0 < trailing_pct < 100:
            raise TradingControlError(
                "Trailing stop 비율은 0%보다 크고 100%보다 작아야 합니다."
            )
        print(f"Trailing stop = 최고가 대비 -{trailing_pct:g}%")
        print("현재가가 계산된 보호가격을 이미 통과했다면 다음 감시에서 즉시 매도될 수 있습니다.")
        print("보유종목별 ATR과 현재가를 조회하고 있습니다...")
        preview = self.controller.preview_bulk_atr_protection(
            trailing_stop_pct=trailing_pct / 100,
        )
        print("\n[손절·익절 일괄 적용 미리보기]")
        print(table(
            ["종목코드", "종목명", "평균단가", "현재가", "ATR", "손절가", "익절가", "Trailing %", "Trailing 가격", "상태", "메시지"],
            [[
                item["ticker"], stock_name(item["ticker"]), money(item["avg_price"]),
                money(item["current_price"]), money(item["atr"]),
                money(item["stop_loss"]), money(item["take_profit"]),
                (
                    f"{item['trailing_stop_pct']:.1%}"
                    if item["trailing_stop_pct"] is not None else "미설정"
                ),
                money(item["trailing_stop"]),
                item["status"], item["message"],
            ] for item in preview],
        ))
        if input("위 값을 전체 적용하려면 APPLY ALL 입력: ").strip().upper() != "APPLY ALL":
            print("일괄 적용을 취소했습니다. 변경된 값은 없습니다.")
            return
        results = self.controller.apply_bulk_atr_protection(preview)
        saved = sum(item["status"] in {"SAVED", "WARNING"} for item in results)
        failed = sum(item["status"] == "ERROR" for item in results)
        print(f"일괄 적용 완료: 저장 {saved}개 / 실패 {failed}개")

    def _scheduler_start(self) -> None:
        print("Scheduler를 시작했습니다." if self.controller.start_scheduler() else "이미 실행 중입니다.")

    def _scheduler_stop(self) -> None:
        print("Scheduler를 중지했습니다." if self.controller.stop_scheduler() else "실행 중이 아닙니다.")

    def _scheduler_toggle(self) -> None:
        if self.controller.scheduler_running:
            print(
                "Scheduler를 중지했습니다."
                if self.controller.stop_scheduler() else "Scheduler 중지에 실패했습니다."
            )
        else:
            print(
                "Scheduler를 시작했습니다."
                if self.controller.start_scheduler() else "Scheduler 시작에 실패했습니다."
            )

    def _toggle_kill(self) -> None:
        current = self.controller.environment()["kill_switch"]
        target = "NORMAL" if current == "HALTED" else "HALTED"
        if input(f"Kill Switch를 {target} 상태로 변경하려면 CONFIRM 입력: ").strip().upper() != "CONFIRM":
            print("변경을 취소했습니다.")
            return
        print(f"Kill Switch: {self.controller.toggle_kill_switch()}")

    def _toggle_ml_filter(self) -> None:
        current = self.controller.ml_filter_enabled()
        target = "OFF" if current else "ON"
        if target == "OFF":
            print("경고: ML 확률과 순위 검사를 건너뛰고 분석 승인 후보를 Risk Node로 전달합니다.")
        if input(f"ML Filter를 {target}으로 변경하려면 CONFIRM 입력: ").strip().upper() != "CONFIRM":
            print("변경을 취소했습니다.")
            return
        enabled = self.controller.toggle_ml_filter()
        print(f"ML Filter: {'ON' if enabled else 'OFF'}")

    def _test_slack(self) -> None:
        if input("Slack 테스트 메시지를 전송하려면 SEND 입력: ").strip().upper() != "SEND":
            print("전송을 취소했습니다.")
            return
        self.controller.test_notification()
        print("Slack 테스트 메시지를 전송했습니다.")

    def _top_recommendations(self) -> None:
        profile = get_market_profile(self.controller.context.config.market_region)
        print("\n[분석 유니버스 선택]")
        print(f"1. {profile.universes[0]}   2. {profile.universes[1]}   3. Both")
        scope = {
            "1": profile.universes[0], "2": profile.universes[1], "3": "BOTH",
        }.get(input("선택 > ").strip())
        if scope is None:
            raise TradingControlError("유니버스 번호를 다시 확인하세요.")
        refresh = input(
            "오늘 저장된 추천이 있으면 사용합니다. 다시 분석하려면 REFRESH 입력: "
        ).strip().upper() == "REFRESH"

        def progress(index: int, total: int, ticker: str) -> None:
            print(f"[{index}/{total}] {ticker} {stock_name(ticker)} 분석 중...", flush=True)

        recommendations = self.controller.top_recommendations(
            universe_scope=scope, refresh=refresh, progress=progress
        )
        if not recommendations:
            print("추천할 종목이 없습니다. 먼저 장전 준비 작업을 확인하세요.")
            return
        print(f"\n[오늘의 Top10 pick - {scope}]")
        print(table(
            ["순위", "종목코드", "종목명", "시장", "섹터", "종합", "기술", "기본", "뉴스", "수급", "ML score", "핵심 근거"],
            [[
                item["rank"], item["ticker"], item["name"], item["market"],
                item["sector"], f"{item['total_score']:.1f}",
                f"{item['technical_score']:.1f}", f"{item['fundamental_score']:.1f}",
                f"{item['news_score']:.1f}", f"{item['flow_score']:.1f}",
                f"{float(item.get('ml_score', 0)):.4f}",
                item["recommendation_reason"],
            ] for item in recommendations],
        ))
        print("\n[추천 근거 상세]")
        detail_rows = []
        for item in recommendations:
            detail_rows.extend([
                [item["rank"], item["ticker"], item["name"], "기술", item["technical_reason"]],
                [item["rank"], item["ticker"], item["name"], "기본", item["fundamental_reason"]],
                [item["rank"], item["ticker"], item["name"], "뉴스", item["news_reason"]],
                [item["rank"], item["ticker"], item["name"], "수급", item["flow_reason"]],
            ])
        print(table(["순위", "종목코드", "종목명", "분석", "추천 근거"], detail_rows))
        print("※ 종합점수는 기술 30%, 기본 25%, 뉴스 20%, 수급 25% 가중치입니다.")
        saved = self.controller.context.store.get_control(
            "top_recommendations_latest"
        ) or {}
        print(
            "PDF 분석 리포트: "
            f"{saved.get('report_url') or saved.get('report_path', '-')}"
        )

    def _review_rebalance_orders(self, orders: list[dict]) -> list[dict] | None:
        """Require an explicit per-security decision before rebalance execution."""
        reviewed = []
        total = len(orders)
        print("\n[종목별 주문 검토]")
        print("각 주문마다 A(승인), M(구분·수량 수정), X(제외), Q(전체 취소)를 입력하세요.")
        for index, item in enumerate(orders, 1):
            print(f"\n[{index}/{total}] {item['ticker']} {item['name']}")
            print(table(
                ["제안", "수량", "현재수량", "예상가격", "예상금액", "근거"],
                [[
                    item["side"], f"{item['quantity']:,}",
                    f"{item.get('original_quantity', 0):,}", money(item["price"]),
                    money(item["estimated_value"]), item["reason"],
                ]],
            ))
            while True:
                choice = input("판단 [A/M/X/Q]: ").strip().upper()
                if choice == "A":
                    reviewed.append(dict(item))
                    break
                if choice == "X":
                    print("이 주문을 실행 대상에서 제외했습니다.")
                    break
                if choice == "Q":
                    return None
                if choice == "M":
                    side = input(
                        f"주문구분 BUY/SELL [{item['side']}]: "
                    ).strip().upper() or item["side"]
                    if side not in {"BUY", "SELL"}:
                        print("BUY 또는 SELL만 입력할 수 있습니다.")
                        continue
                    try:
                        quantity_text = input(
                            f"수량 [{item['quantity']}]: "
                        ).strip()
                        quantity = int(quantity_text or item["quantity"])
                    except ValueError:
                        print("수량은 정수로 입력하세요.")
                        continue
                    if quantity <= 0:
                        print("수량은 1 이상이어야 합니다.")
                        continue
                    reviewed.append({
                        **item, "side": side, "quantity": quantity,
                        "reason": f"{item['reason']} [사용자 수정]",
                    })
                    break
                print("A, M, X 또는 Q 중 하나를 입력하세요.")
        return reviewed

    def _rebalance(self) -> None:
        print("\n보유종목, 10번에서 저장한 오늘의 Top10과 시장뉴스를 종합 분석합니다.")
        print("Top10 종목분석은 다시 실행하지 않고 마지막으로 선택한 유니버스 결과를 사용합니다.")
        print("LLM은 제안만 생성하며 Risk Validator 통과와 사용자 승인이 필요합니다.")

        def progress(index: int, total: int, ticker: str) -> None:
            print(f"[{index}/{total}] {ticker} {stock_name(ticker)} 분석 중...", flush=True)

        package = self.controller.rebalance_proposal(progress=progress)
        proposal = package["proposal"]
        validation = package["validation"]
        news = package["snapshot"]["market_news"]
        print(
            "사용 Top10 유니버스: "
            f"{package['snapshot'].get('recommendation_universe_scope', 'BOTH')}"
        )
        print("\n[오늘의 시장뉴스 요약]")
        print(
            f"뉴스 감성: {news['sentiment']} / "
            f"긍정 {news['positive_hits']} / 부정 {news['negative_hits']}"
        )
        for item in news["headlines"][:5]:
            print(f"- {item.get('title', '-')}")
        print("\n[LLM 시장 판단]")
        print(f"시장 상태: {proposal['market_view']}")
        print(f"권장 현금: {proposal['recommended_cash_pct']:.1f}%")
        print(f"시장 요약: {proposal['market_summary']}")
        print(f"종합 근거: {proposal['overall_reason']}")
        print(f"포트폴리오 진단: {proposal.get('portfolio_assessment') or '-'}")
        print(f"종목뉴스 종합: {proposal.get('news_assessment') or '-'}")
        print(f"\n상세 PDF 리포트: {package.get('report_url') or package.get('report_path', '-')}")
        print("\n[LLM 제안 주문 미리보기]")
        print(table(
            ["구분", "종목코드", "종목명", "현재비중", "목표비중", "수량", "예상가격", "예상금액", "신뢰도", "근거"],
            [[
                item["side"], item["ticker"], item["name"],
                f"{item['current_weight_pct']:.1f}%",
                f"{item['target_weight_pct']:.1f}%", f"{item['quantity']:,}",
                money(item["price"]), money(item["estimated_value"]),
                f"{item['confidence']:.0%}", item["reason"],
            ] for item in validation["orders"]],
        ))
        print(
            f"예상 회전율: {validation['turnover_pct']:.1%} / "
            f"예상 현금비중: {validation['projected_cash_pct']:.1%}"
        )
        print("\n[리밸런싱 제안 검토]")
        print("A. 현재 제안으로 주문별 검토 진행")
        print("R. 사용자 의견을 입력해 LLM 제안 수정")
        print("Q. 저장하고 종료")
        decision = input("선택 [A/R/Q]: ").strip().upper()
        if decision == "R":
            feedback = input("LLM에 전달할 수정 의견: ").strip()
            if not feedback:
                raise TradingControlError("수정 의견을 입력하세요.")
            revised = self.controller.revise_rebalance_proposal(
                package["proposal_id"], feedback
            )
            print(
                f"수정 제안 저장 완료: {revised['proposal_id']} "
                f"(수정 {revised.get('revision', 1)}회)"
            )
            print("수정된 제안을 다시 표시합니다.")
            return self._rebalance()
        if decision == "Q":
            print("현재 제안을 저장한 상태로 종료합니다. 주문은 예약되지 않았습니다.")
            return
        if decision != "A":
            raise TradingControlError("A, R 또는 Q 중 하나를 입력하세요.")

        proposal_id = package["proposal_id"]
        if not validation["orders"]:
            print("검토할 리밸런싱 주문이 없습니다.")
            return
        reviewed_orders = self._review_rebalance_orders(validation["orders"])
        if reviewed_orders is None:
            print("종목별 검토를 취소했습니다. 주문은 발생하지 않았습니다.")
            return
        package = self.controller.review_rebalance_orders(
            proposal_id, reviewed_orders
        )
        validation = package["validation"]
        print("\n[사용자 검토 후 최종 주문]")
        print(table(
            ["구분", "종목코드", "종목명", "수량", "예상가격", "예상금액", "목표비중"],
            [[
                item["side"], item["ticker"], item["name"],
                f"{item['quantity']:,}", money(item["price"]),
                money(item["estimated_value"]), f"{item['target_weight_pct']:.1f}%",
            ] for item in validation["orders"]],
        ))
        print(
            f"재검증 회전율: {validation['turnover_pct']:.1%} / "
            f"재검증 현금비중: {validation['projected_cash_pct']:.1%}"
        )
        override_risk = False
        if validation["errors"]:
            print("\n[Risk Validator 거부 사유]")
            for error in validation["errors"]:
                print(f"- {error}")
            if not validation.get("override_allowed"):
                print("주문 무결성 관련 오류가 포함되어 Override할 수 없습니다.")
                print("주문은 실행되지 않습니다.")
                return
            print("\n경고: 이 제안은 투자정책 한도를 위반합니다.")
            print("Override하면 위 위험을 사용자가 직접 수락하고 실제 주문을 실행합니다.")
            phrase = f"OVERRIDE {proposal_id}"
            entered = input(f"Risk Validator를 Override하려면 '{phrase}' 입력: ").strip()
            if entered != phrase:
                print("리밸런싱 실행을 취소했습니다. 주문은 발생하지 않았습니다.")
                return
            override_risk = True
        if not validation["orders"]:
            print("사용자가 승인한 리밸런싱 주문이 없습니다.")
            return
        if not override_risk:
            phrase = f"REBALANCE {proposal_id}"
            print(f"\n제안서 ID: {proposal_id}")
            entered = input(f"실제 주문을 승인하려면 '{phrase}' 입력: ").strip()
            if entered != phrase:
                print("리밸런싱 실행을 취소했습니다. 주문은 발생하지 않았습니다.")
                return
        result = self.controller.execute_rebalance(
            proposal_id, override_risk=override_risk
        )
        print(f"\n리밸런싱 실행 상태: {result['status']}")
        if result["status"] == "AWAITING_SELL_FILLS":
            print("기존 또는 이번 매도 주문이 아직 체결·취소되지 않아 매수를 보류합니다.")
            print(table(
                ["종목코드", "구분", "상태", "주문수량", "KIS 주문번호", "마지막 갱신"],
                [[
                    item["ticker"], item["side"], item["status"],
                    f"{item['quantity']:,}", item["broker_order_id"],
                    korean_time(item["updated_at"]),
                ] for item in result.get("blocking_orders", [])],
            ))
            print("KIS에서 해당 주문을 체결 또는 취소한 뒤 reconciliation을 실행하고 다시 시도하세요.")
        if result["orders"]:
            print(table(
                ["구분", "종목코드", "종목명", "수량", "가격", "상태", "주문번호"],
                [[
                    item["side"], item["ticker"], item["name"],
                    f"{item['quantity']:,}", money(item["price"]),
                    item["status"], item.get("order_id") or "-",
                ] for item in result["orders"]],
            ))

    def _run_job(self) -> None:
        print("1.pre_open  2.opening_buy  3.monitor  4.post_close  5.reconciliation")
        mapping = {
            "1": "pre_open", "2": "opening_buy", "3": "monitor",
            "4": "post_close", "5": "reconciliation",
        }
        name = mapping.get(input("작업 번호: ").strip())
        if name is None:
            raise TradingControlError("작업 번호가 올바르지 않습니다.")
        result = self.controller.run_job(name)
        print(f"{result.job}: {result.status} {result.error or ''}")

    def _audit(self) -> None:
        events = self.controller.audit_history()
        print(table(["ID", "시각", "이벤트", "대상"], [[
            item["id"], item["created_at"][:19].replace("T", " "),
            item["event_type"], item.get("entity_key") or "-",
        ] for item in events]))


def main() -> None:
    service = create_live_trading_service()
    TradingConsole(TradingController(service)).run()


if __name__ == "__main__":
    main()
