from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from analysis.economy_data import update_economy_data
from ml.sector_data import update_universe_sectors
UNIVERSE_PATH = Path("data/ml/universe_top200.csv")
PANEL_PATH = Path("data/ml/market_panel.csv")
FLOW_HISTORY_PATH = Path("data/ml/flow_history.csv")
FUNDAMENTAL_HISTORY_PATH = Path("data/ml/fundamental_history.csv")
NAVER_MARKET_URL = "https://finance.naver.com/sise/sise_market_sum.naver"


def download_top_market_cap_universe(
    per_market: int = 100,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows = []
    for market, suffix, sosok in (
        ("KOSPI", ".KS", "0"),
        ("KOSDAQ", ".KQ", "1"),
    ):
        page = 1
        while len([row for row in rows if row["market"] == market]) < per_market:
            response = client.get(
                NAVER_MARKET_URL,
                params={"sosok": sosok, "page": page},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            response.raise_for_status()
            response.encoding = "euc-kr"
            soup = BeautifulSoup(response.text, "html.parser")
            page_rows = []
            for anchor in soup.select("a.tltle[href*='code=']"):
                code = anchor["href"].split("code=", 1)[1].split("&", 1)[0]
                page_rows.append({
                    "market": market,
                    "ticker": f"{code}{suffix}",
                    "name": anchor.get_text(strip=True),
                    "market_cap_rank": (page - 1) * 50 + len(page_rows) + 1,
                    "kis_sector": "UNKNOWN",
                    "kis_market": "",
                    "ml_sector": "UNKNOWN",
                    "sector": "UNKNOWN",
                })
            if not page_rows:
                raise ValueError(f"No {market} market-cap rows on page {page}")
            rows.extend(page_rows)
            page += 1

    result = pd.DataFrame(rows).groupby("market", group_keys=False).head(per_market)
    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")
    return result.reset_index(drop=True)


def load_universe(
    refresh: bool = False,
    per_market: int = 100,
    update_sectors: bool = True,
) -> pd.DataFrame:
    if UNIVERSE_PATH.exists() and not refresh:
        universe = pd.read_csv(UNIVERSE_PATH, dtype=str, keep_default_na=False)
    else:
        universe = download_top_market_cap_universe(per_market=per_market)
    if update_sectors:
        universe = update_universe_sectors(universe)
        universe.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")
    return universe


def download_price_panel(
    universe: pd.DataFrame,
    period: str = "5y",
) -> pd.DataFrame:
    tickers = universe["ticker"].tolist()
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    frames = []
    for ticker in tickers:
        try:
            frame = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not set(required).issubset(frame) or frame["Close"].notna().sum() < 100:
            continue
        frame = frame[required].dropna(subset=["Close"])
        frame["ticker"] = ticker
        frames.append(frame.reset_index().rename(columns={"Date": "date"}))
    if not frames:
        raise ValueError("No price history was downloaded for the universe")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def attach_point_in_time_data(
    prices: pd.DataFrame,
    flow_path: Path = FLOW_HISTORY_PATH,
    fundamental_path: Path = FUNDAMENTAL_HISTORY_PATH,
) -> pd.DataFrame:
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    if flow_path.exists():
        flow = pd.read_csv(flow_path, parse_dates=["date"])
        required = {"date", "ticker", "foreign_net", "institution_net"}
        if not required.issubset(flow):
            raise ValueError(f"Flow history needs columns: {sorted(required)}")
        flow["date"] = pd.to_datetime(flow["date"]).astype("datetime64[ns]")
        panel = panel.merge(flow[list(required)], on=["date", "ticker"], how="left")

    if fundamental_path.exists():
        fundamental = pd.read_csv(fundamental_path, parse_dates=["available_date"])
        required = {"available_date", "ticker", "per", "pbr", "psr", "pcr", "ev_ebitda", "roe"}
        if not required.issubset(fundamental):
            raise ValueError(
                f"Fundamental history needs columns: {sorted(required)}"
            )
        fundamental["available_date"] = pd.to_datetime(
            fundamental["available_date"]
        ).astype("datetime64[ns]")
        merged = []
        for ticker, group in panel.groupby("ticker", sort=False):
            history = fundamental[fundamental["ticker"].eq(ticker)].sort_values(
                "available_date"
            )
            if history.empty:
                merged.append(group)
                continue
            history_columns = [column for column in (
                "available_date", "per", "pbr", "psr", "pcr", "ev_ebitda", "roe",
                "earnings_growth",
            ) if column in history]
            merged.append(pd.merge_asof(
                group.sort_values("date"),
                history[history_columns],
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            ).drop(columns="available_date"))
        panel = pd.concat(merged, ignore_index=True)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_or_update_panel(
    refresh_universe: bool = False,
    per_market: int = 100,
    universe_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = load_universe(refresh=refresh_universe, per_market=per_market)
    if universe_limit is not None:
        universe = universe.head(universe_limit).copy()
    prices = attach_point_in_time_data(download_price_panel(universe))
    economy = update_economy_data(prices["date"].min(), prices["date"].max())
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")
    return prices, universe, economy
