from __future__ import annotations

import argparse

import pandas as pd

from analysis.economy_data import update_economy_data
from ml.collect_flow_history import collect_flow_history
from ml.collect_fundamental_history import collect_fundamental_history
from ml.config import DEFAULT_START
from ml.price_history import collect_price_history
from ml.universe_history import KrxAPIError, collect_universe_history


def collect_data(
    start: str = DEFAULT_START,
    end: str | None = None,
    *,
    limit_sessions: int | None = None,
    skip_flow: bool = False,
    skip_fundamental: bool = False,
) -> dict:
    universe = collect_universe_history(start, end, limit_sessions=limit_sessions)
    tickers = sorted(universe["ticker"].unique())
    prices = collect_price_history(tickers, start, end)
    effective_end = pd.Timestamp(end or pd.Timestamp.today())
    years = max(1, round((effective_end - pd.Timestamp(start)).days / 365.25))
    flow = None
    fundamentals = None
    if not skip_flow:
        flow = collect_flow_history(years=years, tickers=tickers)
    if not skip_fundamental:
        fundamentals = collect_fundamental_history(years=years, tickers=tickers)
    economy = update_economy_data(start, end)
    return {
        "universe": universe,
        "prices": prices,
        "flow": flow,
        "fundamentals": fundamentals,
        "economy": economy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect point-in-time universe and price sources")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-fundamental", action="store_true")
    args = parser.parse_args()
    try:
        result = collect_data(
            args.start,
            args.end,
            limit_sessions=args.limit_sessions,
            skip_flow=args.skip_flow,
            skip_fundamental=args.skip_fundamental,
        )
    except KrxAPIError as error:
        parser.exit(
            2,
            f"KRX authorization failed: {error}\n"
            "Run: uv run python -m ml.check_krx_api\n",
        )
    print(
        f"Collected universe={len(result['universe']):,}, "
        f"prices={len(result['prices']):,}, economy={len(result['economy']):,} rows"
    )


if __name__ == "__main__":
    main()
