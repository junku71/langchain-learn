from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from ml.panel_data import FUNDAMENTAL_HISTORY_PATH, load_universe


COLUMNS = [
    "available_date", "ticker", "per", "pbr", "psr", "pcr", "ev_ebitda",
    "roe", "earnings_growth",
]
DART_BASE = "https://opendart.fss.or.kr/api"


def _number(value) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


class DartFundamentalProvider:
    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()
        self._corp_codes: dict[str, str] | None = None

    def corp_codes(self) -> dict[str, str]:
        if self._corp_codes is not None:
            return self._corp_codes
        response = self.session.get(
            f"{DART_BASE}/corpCode.xml",
            params={"crtfc_key": self.api_key},
            timeout=30,
        )
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            root = ElementTree.fromstring(archive.read("CORPCODE.xml"))
        self._corp_codes = {
            (node.findtext("stock_code") or "").strip(): node.findtext("corp_code") or ""
            for node in root.findall("list")
            if (node.findtext("stock_code") or "").strip()
        }
        return self._corp_codes

    def annual_statement(self, ticker: str, year: int) -> tuple[pd.Timestamp, list[dict]] | None:
        code = ticker.split(".", 1)[0]
        corp_code = self.corp_codes().get(code)
        if not corp_code:
            return None
        for fs_div in ("CFS", "OFS"):
            response = self.session.get(
                f"{DART_BASE}/fnlttSinglAcntAll.json",
                params={
                    "crtfc_key": self.api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": "11011",
                    "fs_div": fs_div,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "000" and payload.get("list"):
                rows = payload["list"]
                receipt = str(rows[0].get("rcept_no", ""))[:8]
                try:
                    available_date = pd.to_datetime(receipt, format="%Y%m%d")
                except ValueError:
                    continue
                return available_date, rows
        return None


def _account(rows: list[dict], ids: tuple[str, ...], names: tuple[str, ...] = ()) -> float | None:
    for row in rows:
        account_id = str(row.get("account_id", ""))
        account_name = str(row.get("account_nm", "")).replace(" ", "")
        if account_id in ids or any(name in account_name for name in names):
            value = _number(row.get("thstrm_amount"))
            if value is not None:
                return value
    return None


def _first_account(rows: list[dict], ids: tuple[str, ...]) -> float | None:
    return _account(rows, ids)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return np.nan
    return numerator / denominator


def _market_data(ticker: str, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, tuple[float, float]]:
    if not dates:
        return {}
    stock = yf.Ticker(ticker)
    start = min(dates) - pd.Timedelta(days=10)
    end = max(dates) + pd.Timedelta(days=10)
    prices = stock.history(start=start, end=end, auto_adjust=False)["Close"]
    shares = stock.get_shares_full(start=start, end=end)
    if prices.empty or shares is None or shares.empty:
        return {}
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    shares.index = pd.to_datetime(shares.index).tz_localize(None).normalize()
    result = {}
    for available_date in dates:
        prior_price = prices.loc[prices.index <= available_date]
        prior_shares = shares.loc[shares.index <= available_date]
        if not prior_price.empty and not prior_shares.empty:
            result[available_date] = (float(prior_price.iloc[-1]), float(prior_shares.iloc[-1]))
    return result


def collect_fundamental_history(
    years: int = 5,
    output_path: Path = FUNDAMENTAL_HISTORY_PATH,
    limit: int | None = None,
    provider: DartFundamentalProvider | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("DART_API_KEY", "")
    if provider is None and not api_key:
        raise ValueError("DART_API_KEY is required in .env")
    provider = provider or DartFundamentalProvider(api_key)
    universe = (
        pd.DataFrame({"ticker": tickers})
        if tickers is not None
        else load_universe(update_sectors=False)
    )
    if limit:
        universe = universe.head(limit)
    current_year = pd.Timestamp.today().year
    records = []

    for index, ticker in enumerate(universe["ticker"], start=1):
        statements = []
        for year in range(current_year - years - 1, current_year):
            statement = provider.annual_statement(ticker, year)
            if statement:
                statements.append((year, *statement))
        market = _market_data(ticker, [item[1] for item in statements])
        for year, available_date, rows in statements:
            revenue = _account(rows, ("ifrs-full_Revenue", "ifrs-full_RevenueFromContractsWithCustomers"), ("매출액", "영업수익"))
            net_income = _account(rows, ("ifrs-full_ProfitLoss",), ("당기순이익",))
            equity = _account(rows, ("ifrs-full_Equity",), ("자본총계",))
            operating_cash = _account(rows, ("ifrs-full_CashFlowsFromUsedInOperatingActivities",), ("영업활동현금흐름",))
            operating_income = _account(rows, ("dart_OperatingIncomeLoss",), ("영업이익",))
            depreciation = _account(rows, (), ("감가상각비", "감가상각및무형자산상각비"))
            debt = _account(rows, (), ("차입금", "사채"))
            cash = _account(rows, ("ifrs-full_CashAndCashEquivalents",), ("현금및현금성자산",))
            revenue = revenue or _first_account(rows, (
                "ifrs-full_RevenueFromContractsWithCustomers", "ifrs-full_Revenue",
            ))
            net_income = net_income or _first_account(rows, (
                "ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs-full_ProfitLoss",
            ))
            operating_income = operating_income or _first_account(rows, (
                "dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities",
            ))
            depreciation = depreciation or _first_account(rows, (
                "ifrs-full_DepreciationAndAmortisationExpense",
                "dart_AdjustmentsForDepreciationExpense",
            ))
            debt = debt or _first_account(rows, (
                "ifrs-full_ShorttermBorrowings", "ifrs-full_LongtermBorrowings",
            ))
            market_values = market.get(available_date)
            market_cap = market_values[0] * market_values[1] if market_values else None
            ebitda = (operating_income + (depreciation or 0)) if operating_income is not None else None
            enterprise_value = (
                market_cap + (debt or 0) - (cash or 0)
                if market_cap is not None
                else None
            )
            records.append({
                "available_date": available_date,
                "ticker": ticker,
                "per": _safe_ratio(market_cap, net_income),
                "pbr": _safe_ratio(market_cap, equity),
                "psr": _safe_ratio(market_cap, revenue),
                "pcr": _safe_ratio(market_cap, operating_cash),
                "ev_ebitda": _safe_ratio(enterprise_value, ebitda),
                "roe": _safe_ratio(net_income, equity) * 100,
                "_net_income": net_income,
            })
        print(f"[{index}/{len(universe)}] {ticker}: {len(statements)} annual filings")

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["ticker", "available_date"])
        result["earnings_growth"] = result.groupby("ticker")["_net_income"].pct_change(fill_method=None)
        result = result.drop(columns="_net_income")
    result = result.reindex(columns=COLUMNS)
    if output_path.exists():
        old = pd.read_csv(output_path, parse_dates=["available_date"], dtype={"ticker": str})
        result = pd.concat([old.reindex(columns=COLUMNS), result], ignore_index=True)
    result = result.drop_duplicates(["available_date", "ticker"], keep="last")
    result = result.sort_values(["available_date", "ticker"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect point-in-time DART fundamentals")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = collect_fundamental_history(years=args.years, limit=args.limit)
    print(f"Saved {len(result):,} rows to {FUNDAMENTAL_HISTORY_PATH}")


if __name__ == "__main__":
    main()
