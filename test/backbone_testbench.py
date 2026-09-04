import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.ticker_mapper import get_company_name, get_yfinance_ticker


def resolve_domestic_stock(stock_name_or_ticker: str) -> tuple[str, str]:
    value = stock_name_or_ticker.strip()
    upper = value.upper()
    if not value:
        raise ValueError("Stock name is required")

    if re.fullmatch(r"\d{6}", upper):
        candidates = [f"{upper}.KS", f"{upper}.KQ"]
        for ticker in candidates:
            name = get_company_name(ticker)
            if name != upper:
                return ticker, name
        raise KeyError(f"Unknown domestic stock code: {value}")

    if re.fullmatch(r"\d{6}\.(KS|KQ)", upper):
        name = get_company_name(upper)
        if name == upper.split(".", 1)[0]:
            raise KeyError(f"Unknown domestic ticker: {value}")
        return upper, name

    for market in ("KOSPI", "KOSDAQ"):
        try:
            ticker = get_yfinance_ticker(value, market)
            return ticker, get_company_name(ticker, market)
        except KeyError:
            continue
    raise KeyError(f"Domestic stock name not found: {value}")


def create_initial_state(
    ticker: str,
    sector: str = "UNKNOWN",
    account_size: float = 50_000_000,
    risk_per_trade: float = 0.01,
    trailing_stop_pct: float = 0.08,
) -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "account_size": account_size,
        "risk_per_trade": risk_per_trade,
        "trailing_stop_pct": trailing_stop_pct,
        "market_data": None,
        "technical_result": None,
        "fundamental_result": None,
        "news_data": None,
        "news_result": None,
        "flow_result": None,
        "merged_result": None,
        "ml_result": None,
        "risk_result": None,
        "final_decision": None,
        "decision_result": None,
        "agent_errors": None,
        "paper_order_result": None,
        "portfolio_result": None,
        "portfolio_guard_result": None,
    }


def _print_value(name: str, value) -> None:
    if isinstance(value, pd.DataFrame):
        start = value.index.min().date() if not value.empty else None
        end = value.index.max().date() if not value.empty else None
        print(f"  {name}: DataFrame(rows={len(value)}, {start}..{end})")
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, default=str)
        print(f"  {name}: {text[:2000]}{'...' if len(text) > 2000 else ''}")
    else:
        text = str(value)
        print(f"  {name}: {text[:1000]}{'...' if len(text) > 1000 else ''}")


def run_backbone(initial_state: dict) -> dict:
    from multiagent_graph import graph

    completed = []
    final_state = initial_state.copy()
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node, update in event.items():
            completed.append(node)
            print(f"\n[{len(completed):02d}] {node}")
            if isinstance(update, dict):
                final_state.update(update)
                for name, value in update.items():
                    _print_value(name, value)
            else:
                _print_value("result", update)

    print("\n=== Backbone Summary ===")
    print("Flow:", " -> ".join(completed))
    print("Decision:", final_state.get("final_decision"))
    ml_result = final_state.get("ml_result") or {}
    if ml_result:
        print(f"ML probability: {ml_result.get('up_probability', 0):.2%}")
        print(f"Model reused: {ml_result.get('model_reused')}")
    return final_state


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run the complete domestic-stock LangGraph backbone."
    )
    parser.add_argument("stock", nargs="?", help="Korean stock name, code, or ticker")
    parser.add_argument("--sector", default="UNKNOWN")
    parser.add_argument("--account-size", type=float, default=50_000_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--trailing-stop", type=float, default=0.08)
    args = parser.parse_args()

    stock = args.stock or input("국내 종목명: ").strip()
    ticker, company_name = resolve_domestic_stock(stock)
    print(f"Resolved: {company_name} -> {ticker}")
    state = create_initial_state(
        ticker=ticker,
        sector=args.sector,
        account_size=args.account_size,
        risk_per_trade=args.risk_per_trade,
        trailing_stop_pct=args.trailing_stop,
    )
    run_backbone(state)


if __name__ == "__main__":
    main()
