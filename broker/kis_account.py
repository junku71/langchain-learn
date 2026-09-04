import sys

import requests
from dotenv import load_dotenv

from broker.kis import KISAPIError, KISBroker


def get_kis_account_report(
    broker: KISBroker,
) -> dict:

    balance = broker.get_balance()
    positions = broker.get_positions()
    position_details = {}

    for ticker, position in positions.items():
        current_price = broker.get_current_price(ticker)
        market_value = current_price * position.quantity
        cost_value = position.avg_price * position.quantity
        unrealized_pnl = market_value - cost_value

        position_details[ticker] = {
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            "current_price": current_price,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pct": (
                unrealized_pnl / cost_value
                if cost_value > 0
                else 0.0
            ),
        }

    return {
        "balance": balance,
        "positions": position_details,
    }


def print_kis_account_report(
    report: dict,
) -> None:

    balance = report["balance"]
    positions = report["positions"]

    print("\n=== 한국투자증권 계좌 현황 ===\n")
    print(f"예수금: {balance['cash']:,.0f}원")
    print(f"총 평가금액: {balance['total_equity']:,.0f}원")
    print(f"평가손익: {balance['unrealized_pnl']:+,.0f}원")
    print(f"보유종목 수: {len(positions)}개")

    if not positions:
        print("\n보유종목이 없습니다.")
        return

    print("\n=== 보유종목 ===")

    for ticker, position in positions.items():
        print(f"\n[{ticker}]")
        print(f"수량: {position['quantity']:,}주")
        print(f"평균단가: {position['avg_price']:,.0f}원")
        print(f"현재가: {position['current_price']:,.0f}원")
        print(f"평가금액: {position['market_value']:,.0f}원")
        print(f"평가손익: {position['unrealized_pnl']:+,.0f}원")
        print(f"수익률: {position['unrealized_pct']:+.2%}")


def main() -> int:
    load_dotenv()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        broker = KISBroker.from_env()
        report = get_kis_account_report(broker)
    except (
        KISAPIError,
        requests.RequestException,
        ValueError,
    ) as error:
        print(f"KIS 계좌 조회 실패: {error}", file=sys.stderr)
        return 1

    print_kis_account_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
