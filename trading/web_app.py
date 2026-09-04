from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

# `streamlit run trading/web_app.py` places the script directory (`trading`)
# on sys.path, not necessarily the repository root. Bootstrap the root before
# importing the local packages so the documented command works consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from trading.controller import TradingControlError, TradingController
from trading.display import stock_name
from trading.factory import create_live_trading_service


st.set_page_config(page_title="KRX 자동매매", page_icon="📈", layout="wide")


@st.cache_resource
def controller() -> TradingController:
    return TradingController(create_live_trading_service())


def run(action, success: str | None = None):
    try:
        result = action()
        if success:
            st.success(success)
        return result
    except TradingControlError as error:
        st.error(str(error))
    except Exception as error:
        st.exception(error)
    return None


def money(value) -> str:
    try:
        return f"{float(value):,.0f}원"
    except (TypeError, ValueError):
        return "-"


def order_rows(orders: list[dict]) -> list[dict]:
    rows = []
    for item in orders:
        payload = item.get("payload", {})
        price, quantity = payload.get("price"), payload.get("quantity")
        try:
            amount = float(price) * int(quantity)
        except (TypeError, ValueError):
            amount = None
        rows.append({
            "시각": item.get("updated_at"), "종목코드": item.get("ticker"),
            "종목명": stock_name(item.get("ticker", "")), "구분": item.get("side"),
            "상태": item.get("status"), "가격": price, "수량": quantity,
            "총주문금액": amount, "KIS 주문번호": item.get("broker_order_id") or "-",
            "idempotency_key": item.get("idempotency_key"),
        })
    return rows


def position_rows(ctrl: TradingController) -> tuple[list[dict], dict]:
    balance, positions = ctrl.account_snapshot()
    rows = []
    for ticker, position in positions.items():
        protection = ctrl.context.store.get_protection(ticker)
        try:
            current = ctrl.quote(ticker)
        except Exception:
            current = None
        pnl = (
            (current - position.avg_price) * position.quantity
            if current is not None else None
        )
        return_pct = (
            (current / position.avg_price - 1) * 100
            if current is not None and position.avg_price else None
        )
        rows.append({
            "종목코드": ticker, "종목명": stock_name(ticker), "수량": position.quantity,
            "평균단가": position.avg_price, "현재가": current, "수익률(%)": return_pct,
            "손익금액": pnl, "손절가": (protection.stop_loss if protection else position.stop_loss),
            "익절가": (protection.take_profit if protection else position.take_profit),
            "Trailing(%)": (
                (protection.trailing_stop_pct if protection else position.trailing_stop_pct) * 100
                if (protection.trailing_stop_pct if protection else position.trailing_stop_pct) is not None
                else None
            ),
        })
    return rows, balance


def status_bar(ctrl: TradingController) -> None:
    env = ctrl.environment()
    columns = st.columns(7)
    values = [
        ("Broker", f"{env['broker']} / {env['account_type']}"),
        ("시장", env["market_phase"]), ("Scheduler", env["scheduler"]),
        ("신규 매수", "ON" if env["buy_enabled"] else "OFF"),
        ("ML Filter", env["ml_filter"]), ("Slack", env["slack"]),
        ("Kill Switch", env["kill_switch"]),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    if env["account_type"] == "REAL" and not env["dry_run"]:
        st.error("REAL ACCOUNT · 실제 주문 모드입니다.")
    elif not env["dry_run"]:
        st.warning("Dry Run이 꺼져 있습니다. 주문은 Broker로 전송됩니다.")


def dashboard(ctrl: TradingController) -> None:
    st.subheader("계좌 대시보드")
    rows, balance = position_rows(ctrl)
    cols = st.columns(4)
    cols[0].metric("예수금", money(balance.get("cash")))
    cols[1].metric("총평가금액", money(balance.get("total_equity")))
    cols[2].metric("평가손익", money(balance.get("unrealized_pnl")))
    cols[3].metric("보유종목", f"{len(rows)}개")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("보유종목이 없습니다.")


def portfolio(ctrl: TradingController) -> None:
    st.subheader("보유종목 및 보호가격")
    rows, _ = position_rows(ctrl)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    positions = ctrl.context.broker.get_positions()
    if not positions:
        return
    with st.expander("종목별 손절·익절·Trailing 변경"):
        ticker = st.selectbox("보유종목", list(positions), format_func=lambda x: f"{x} {stock_name(x)}")
        current = ctrl.quote(ticker)
        position = positions[ticker]
        st.caption(f"평균단가 {money(position.avg_price)} · 현재가 {money(current)}")
        c1, c2, c3 = st.columns(3)
        stop = c1.number_input("손절가", min_value=0.0, value=0.0)
        target = c2.number_input("익절가", min_value=0.0, value=0.0)
        trailing = c3.number_input("Trailing(%)", min_value=0.0, max_value=99.9, value=8.0)
        if st.button("보호가격 저장", type="primary"):
            run(lambda: ctrl.set_protection(
                ticker, stop_loss=stop or None, take_profit=target or None,
                trailing_stop_pct=trailing / 100 if trailing else None,
            ), "보호가격을 저장했습니다.")
    with st.expander("전 종목 -3×ATR / +20% / Trailing 일괄 적용"):
        trailing = st.number_input("일괄 Trailing(%)", 0.1, 99.9, 8.0, key="bulk_trailing")
        if st.button("일괄 적용 미리보기"):
            preview = run(lambda: ctrl.preview_bulk_atr_protection(trailing / 100))
            if preview is not None:
                st.session_state["bulk_preview"] = preview
        preview = st.session_state.get("bulk_preview")
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
            phrase = st.text_input("적용하려면 APPLY ALL 입력")
            if st.button("전 종목 보호가격 적용", type="primary", disabled=phrase != "APPLY ALL"):
                run(lambda: ctrl.apply_bulk_atr_protection(preview), "일괄 적용했습니다.")


def orders(ctrl: TradingController) -> None:
    st.subheader("주문 및 체결")
    c1, c2 = st.columns([1, 4])
    if c1.button("API reconciliation", type="primary"):
        report = run(ctrl.api_reconciliation)
        if report:
            c2.success(
                f"조회 {report['checked']} · 변경 {report['changed']} · "
                f"알림 {report['notified']} · 오류 {report['errors']}"
            )
    pending = ctrl.pending_orders()
    st.markdown("#### 미체결 주문")
    pending_rows = order_rows(pending)
    st.dataframe(pd.DataFrame(pending_rows).drop(columns=["idempotency_key"], errors="ignore"), use_container_width=True, hide_index=True)
    if pending:
        selected = st.selectbox(
            "취소할 주문", range(len(pending)),
            format_func=lambda i: f"{pending[i]['ticker']} · {pending[i]['broker_order_id']} · {pending[i]['status']}",
        )
        intent = pending[selected]
        required = f"CANCEL {intent['broker_order_id']}"
        phrase = st.text_input(f"확인 문구: {required}")
        if st.button("미체결 잔량 전체 취소", type="primary", disabled=phrase != required):
            result = run(lambda: ctrl.cancel_pending_order(intent["idempotency_key"]))
            if result:
                st.success(f"취소 요청: {result['request']['status']}")
    st.markdown("#### 최근 주문 내역")
    history = order_rows(ctrl.order_history())
    st.dataframe(pd.DataFrame(history).drop(columns=["idempotency_key"], errors="ignore"), use_container_width=True, hide_index=True)


def manual_trading(ctrl: TradingController) -> None:
    st.subheader("사용자 지정 주문")
    buy_tab, sell_tab = st.tabs(["매수", "매도"])
    with buy_tab:
        with st.form("manual_buy"):
            ticker = st.text_input("종목코드", placeholder="005930.KS").strip().upper()
            quantity = st.number_input("수량", min_value=1, step=1)
            price = st.number_input("지정가", min_value=0.0, step=100.0)
            sector = st.text_input("섹터", value="UNKNOWN")
            c1, c2, c3 = st.columns(3)
            stop = c1.number_input("손절가", min_value=0.0)
            target = c2.number_input("익절가", min_value=0.0)
            trailing = c3.number_input("Trailing(%)", 0.0, 99.9, 8.0)
            required = f"BUY {ticker.split('.')[0]}" if ticker else "BUY 종목코드"
            confirm = st.text_input(f"주문 확인 문구: {required}")
            submitted = st.form_submit_button("매수 주문", type="primary")
        if submitted:
            if confirm != required:
                st.error("확인 문구가 일치하지 않습니다.")
            else:
                result = run(lambda: ctrl.manual_buy(
                    ticker, int(quantity), limit_price=price or None, sector=sector,
                    stop_loss=stop or None, take_profit=target or None,
                    trailing_stop_pct=trailing / 100 if trailing else None,
                ))
                if result:
                    st.success(f"{result.status} · 주문번호 {result.order_id or '-'}")
    with sell_tab:
        positions = ctrl.context.broker.get_positions()
        if not positions:
            st.info("매도할 보유종목이 없습니다.")
        else:
            with st.form("manual_sell"):
                ticker = st.selectbox("보유종목", list(positions), format_func=lambda x: f"{x} {stock_name(x)}")
                quantity = st.number_input("매도 수량", 1, positions[ticker].quantity, 1)
                price = st.number_input("매도 지정가", min_value=0.0, step=100.0)
                required = f"SELL {ticker.split('.')[0]}"
                confirm = st.text_input(f"주문 확인 문구: {required}")
                submitted = st.form_submit_button("매도 주문", type="primary")
            if submitted:
                if confirm != required:
                    st.error("확인 문구가 일치하지 않습니다.")
                else:
                    result = run(lambda: ctrl.manual_sell(ticker, int(quantity), limit_price=price or None))
                    if result:
                        st.success(f"{result.status} · 주문번호 {result.order_id or '-'}")


def automation(ctrl: TradingController) -> None:
    st.subheader("자동매매 제어")
    env = ctrl.environment()
    c1, c2, c3 = st.columns(3)
    if c1.button("Scheduler 시작", disabled=ctrl.scheduler_running):
        run(ctrl.start_scheduler, "Scheduler를 시작했습니다.")
    if c2.button("Scheduler 중지", disabled=not ctrl.scheduler_running):
        run(ctrl.stop_scheduler, "Scheduler를 중지했습니다.")
    if c3.button("ML Filter ON/OFF"):
        enabled = run(ctrl.toggle_ml_filter)
        if enabled is not None:
            st.success(f"ML Filter {'ON' if enabled else 'OFF'}")
    st.markdown("#### Kill Switch")
    target = "NORMAL" if env["kill_switch"] == "HALTED" else "HALTED"
    phrase = st.text_input(f"{target} 전환 확인 문구: CONFIRM", key="kill_confirm")
    if st.button(f"Kill Switch → {target}", type="primary", disabled=phrase != "CONFIRM"):
        run(ctrl.toggle_kill_switch, f"Kill Switch를 {target}로 변경했습니다.")
    st.markdown("#### 자동 작업 수동 실행")
    job = st.selectbox("작업", ["pre_open", "opening_buy", "monitor", "reconciliation", "post_close"])
    if st.button("선택 작업 실행"):
        result = run(lambda: ctrl.run_job(job))
        if result:
            st.json(asdict(result))
    if st.button("Slack 연결 테스트"):
        run(ctrl.test_notification, "Slack 테스트 메시지를 전송했습니다.")


def analysis_and_rebalance(ctrl: TradingController) -> None:
    st.subheader("종합분석 및 LLM 리밸런싱")
    refresh = st.checkbox("Top10 외부 데이터 새로 분석")
    if st.button("오늘의 Top10 pick", type="primary"):
        with st.spinner("기술·기본·뉴스·수급 분석 중..."):
            items = run(lambda: ctrl.top_recommendations(refresh=refresh))
        if items is not None:
            st.session_state["top10"] = items
    if st.session_state.get("top10"):
        st.dataframe(pd.DataFrame(st.session_state["top10"]), use_container_width=True, hide_index=True)
    st.divider()
    if st.button("LLM 리밸런싱 제안 생성"):
        with st.spinner("보유종목·추천종목 뉴스와 포트폴리오를 종합 분석 중..."):
            package = run(ctrl.rebalance_proposal)
        if package:
            st.session_state["rebalance_package"] = package
    package = st.session_state.get("rebalance_package")
    if not package:
        return
    proposal, validation = package["proposal"], package["validation"]
    st.markdown(f"#### 제안서 `{package['proposal_id']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("시장 상태", proposal["market_view"])
    c2.metric("권장 현금", f"{proposal['recommended_cash_pct']:.1f}%")
    c3.metric("Risk Validator", "통과" if validation["approved"] else "거부")
    st.write(proposal["market_summary"])
    st.info(proposal["overall_reason"])
    report_url = package.get("report_url")
    if report_url:
        st.link_button(
            f"OneDrive PDF 제안서 열기 · {Path(package['report_path']).name}",
            report_url,
        )
    else:
        st.code(package.get("report_path", ""))
    st.dataframe(pd.DataFrame(validation["orders"]), use_container_width=True, hide_index=True)
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)
    override = not validation["approved"] and validation.get("override_allowed", False)
    required = f"{'OVERRIDE' if override else 'REBALANCE'} {package['proposal_id']}"
    confirm = st.text_input(f"실제 주문 승인 문구: {required}")
    disabled = confirm != required or not validation["orders"] or (
        not validation["approved"] and not validation.get("override_allowed", False)
    )
    if st.button("리밸런싱 주문 실행", type="primary", disabled=disabled):
        result = run(lambda: ctrl.execute_rebalance(package["proposal_id"], override_risk=override))
        if result:
            st.success(f"실행 상태: {result['status']}")
            st.dataframe(pd.DataFrame(result.get("orders", [])), use_container_width=True, hide_index=True)


def audit(ctrl: TradingController) -> None:
    st.subheader("감사 로그")
    st.dataframe(pd.DataFrame(ctrl.audit_history()), use_container_width=True, hide_index=True)


def main() -> None:
    ctrl = controller()
    st.title("📈 KRX 자동매매 제어센터")
    status_bar(ctrl)
    page = st.sidebar.radio(
        "메뉴",
        ["대시보드", "포트폴리오", "주문·체결", "수동 주문", "자동매매", "Top10·리밸런싱", "감사 로그"],
    )
    st.sidebar.caption("화면을 새로고침해도 Controller와 Scheduler 상태는 유지됩니다.")
    pages = {
        "대시보드": dashboard, "포트폴리오": portfolio, "주문·체결": orders,
        "수동 주문": manual_trading, "자동매매": automation,
        "Top10·리밸런싱": analysis_and_rebalance, "감사 로그": audit,
    }
    pages[page](ctrl)


if __name__ == "__main__":
    main()
