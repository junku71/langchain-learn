from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import asdict
from datetime import datetime

from trading.config import LiveTradingConfig
from trading.display import stock_name
from trading.factory import create_live_trading_service


def _money(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    def display_width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in value
        )

    text_rows = [["-" if value is None else str(value) for value in row] for row in rows]
    widths = [
        max(display_width(header), *(display_width(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: list[str]) -> str:
        return " | ".join(
            value + " " * (widths[index] - display_width(value))
            for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in text_rows)])


def _print_status_table(broker_name: str, dry_run: bool, balance: dict, positions: dict) -> None:
    print(f"Broker: {broker_name}    Dry run: {dry_run}")
    print()
    print("[잔고]")
    print(
        _table(
            ["예수금", "총평가금액", "평가손익", "보유종목 수"],
            [[
                _money(balance.get("cash")),
                _money(balance.get("total_equity")),
                _money(balance.get("unrealized_pnl")),
                balance.get("position_count", len(positions)),
            ]],
        )
    )
    print()
    print("[보유종목]")
    if not positions:
        print("보유종목이 없습니다.")
        return
    print(
        _table(
            ["종목코드", "종목명", "수량", "평균단가", "섹터", "손절가", "익절가", "트레일링"],
            [
                [
                    ticker,
                    stock_name(ticker),
                    f"{position.quantity:,}",
                    _money(position.avg_price),
                    position.sector or "-",
                    _money(position.stop_loss),
                    _money(position.take_profit),
                    (
                        f"{position.trailing_stop_pct:.1%}"
                        if position.trailing_stop_pct is not None
                        else "-"
                    ),
                ]
                for ticker, position in positions.items()
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live trading scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run continuously")
    status = subparsers.add_parser(
        "status", help="Show broker balance and open positions"
    )
    status.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    once = subparsers.add_parser("once", help="Run jobs due at one timestamp")
    once.add_argument("--at", help="ISO timestamp; defaults to now in Asia/Seoul")
    args = parser.parse_args()

    config = LiveTradingConfig.from_env()
    service = create_live_trading_service(config)
    if args.command == "status":
        broker = service.context.broker
        positions = broker.get_positions()
        balance = broker.get_balance()
        if args.json:
            print(
                json.dumps(
                    {
                        "broker": type(broker).__name__,
                        "dry_run": config.dry_run,
                        "balance": balance,
                        "positions": {
                            ticker: asdict(position)
                            for ticker, position in positions.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        else:
            _print_status_table(
                type(broker).__name__, config.dry_run, balance, positions
            )
        return

    if args.command == "run":
        service.run_forever()
        return

    now = (
        datetime.fromisoformat(args.at).astimezone(config.timezone)
        if args.at
        else datetime.now(config.timezone)
    )
    results = service.run_due(now)
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
