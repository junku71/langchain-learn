from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from analysis.economy_data import EcosRateProvider, YahooMarketProvider
from broker.kis import KISBroker
from ml.collect_fundamental_history import DartFundamentalProvider
from ml.panel_data import FLOW_HISTORY_PATH, FUNDAMENTAL_HISTORY_PATH


def _result(source: str, ok: bool, detail: str) -> dict:
    return {"source": source, "ok": ok, "detail": detail}


def _safe_check(source: str, function) -> dict:
    try:
        return _result(source, True, str(function()))
    except Exception as error:  # A diagnostic must report every source in one run.
        return _result(source, False, f"{type(error).__name__}: {error}")


def _check_yahoo_price(ticker: str, start: date, end: date) -> str:
    frame = yf.download(
        ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True, progress=False,
    )
    if frame.empty:
        raise ValueError("empty OHLCV response")
    return f"rows={len(frame)}, last={pd.Timestamp(frame.index.max()).date()}"


def _check_yahoo_market(start: date, end: date) -> str:
    frame = YahooMarketProvider({
        "KOSPI": "^KS11", "NASDAQ": "^IXIC", "SOX": "^SOX",
        "USD_KRW": "KRW=X", "GOLD": "GC=F", "WTI": "CL=F",
    }).fetch(start, end)
    counts = frame.notna().sum().to_dict()
    if not counts or max(counts.values(), default=0) == 0:
        raise ValueError("empty market response")
    return f"rows={len(frame)}, non_null={counts}"


def _check_ecos(start: date, end: date) -> str:
    provider = EcosRateProvider.from_env()
    if provider is None:
        raise ValueError("ECOS_API_KEY is missing")
    frame = provider.fetch(start, end)
    counts = frame.notna().sum().to_dict()
    if not counts or max(counts.values(), default=0) == 0:
        raise ValueError("empty ECOS response")
    return f"rows={len(frame)}, non_null={counts}"


def _check_kis_flow(ticker: str) -> str:
    rows = KISBroker.from_env().get_investor_flow(ticker)
    if not rows:
        raise ValueError("empty KIS investor response")
    return f"rows={len(rows)}, fields={len(rows[0])}"


def _check_kis_fundamental(ticker: str) -> str:
    values = KISBroker.from_env().get_fundamental_data(ticker)
    available = sorted(key for key, value in values.items() if value is not None)
    if len(available) <= 1:
        raise ValueError("no KIS fundamental values")
    return f"available={available}"


def _check_dart(ticker: str) -> str:
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DART_API_KEY is missing")
    provider = DartFundamentalProvider(api_key)
    current_year = date.today().year
    for year in range(current_year - 1, current_year - 4, -1):
        statement = provider.annual_statement(ticker, year)
        if statement:
            available_date, rows = statement
            return f"year={year}, filing={available_date.date()}, rows={len(rows)}"
    raise ValueError("no annual filing in the last three years")


def _check_csv(path: Path, required: set[str]) -> str:
    if not path.exists():
        raise ValueError("not created yet")
    frame = pd.read_csv(path)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing columns={sorted(missing)}")
    if frame.empty:
        raise ValueError("schema exists but the file has no data rows")
    date_column = "date" if "date" in frame else "available_date"
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return (
        f"rows={len(frame)}, tickers={frame['ticker'].nunique()}, "
        f"range={dates.min().date()}..{dates.max().date()}"
    )


def check_data_sources(ticker: str = "005930.KS", days: int = 45) -> list[dict]:
    load_dotenv()
    end = date.today()
    start = end - timedelta(days=days)
    return [
        _safe_check("Yahoo OHLCV", lambda: _check_yahoo_price(ticker, start, end)),
        _safe_check("Yahoo market/macro", lambda: _check_yahoo_market(start, end)),
        _safe_check("ECOS rates", lambda: _check_ecos(start, end)),
        _safe_check("KIS current flow", lambda: _check_kis_flow(ticker)),
        _safe_check("KIS fundamentals", lambda: _check_kis_fundamental(ticker)),
        _safe_check("DART filings", lambda: _check_dart(ticker)),
        _safe_check("Flow CSV", lambda: _check_csv(
            FLOW_HISTORY_PATH, {"date", "ticker", "foreign_net", "institution_net"}
        )),
        _safe_check("Fundamental CSV", lambda: _check_csv(
            FUNDAMENTAL_HISTORY_PATH,
            {"available_date", "ticker", "per", "pbr", "ev_ebitda", "roe"},
        )),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check all ML sources except KRX")
    parser.add_argument("--ticker", default="005930.KS")
    parser.add_argument("--days", type=int, default=45)
    args = parser.parse_args()
    results = check_data_sources(args.ticker, args.days)
    for result in results:
        mark = "PASS" if result["ok"] else "FAIL"
        print(f"[{mark}] {result['source']}: {result['detail']}")
    passed = sum(result["ok"] for result in results)
    print(f"Summary: {passed}/{len(results)} checks passed (KRX excluded)")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
