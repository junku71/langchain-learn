from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from ml.config import DEFAULT_START, PRICE_HISTORY_PATH, UNIVERSE_HISTORY_PATH
from ml.storage import merge_checkpoint, read_frame


def collect_price_history(
    tickers: list[str] | None = None,
    start: str | date = DEFAULT_START,
    end: str | date | None = None,
) -> pd.DataFrame:
    if tickers is None:
        universe = read_frame(UNIVERSE_HISTORY_PATH)
        if universe.empty:
            raise ValueError("Collect point-in-time universe before price history")
        tickers = sorted(universe["ticker"].astype(str).unique())
    existing = read_frame(PRICE_HISTORY_PATH)
    start_date = pd.Timestamp(start).date()
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        last_by_ticker = existing.groupby("ticker")["date"].max()
        if set(tickers).issubset(last_by_ticker.index):
            start_date = max(start_date, (last_by_ticker.min() + pd.Timedelta(days=1)).date())
    end_date = pd.Timestamp(end or pd.Timestamp.today()).date()
    if start_date > end_date:
        return existing.reset_index(drop=True)
    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
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
        if not set(required).issubset(frame):
            continue
        frame = frame[required].dropna(subset=["Close"])
        frame["ticker"] = ticker
        frame = frame.reset_index().rename(columns={"Date": "date"})
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).astype("datetime64[ns]")
        frames.append(frame)
    if not frames:
        return existing
    return merge_checkpoint(pd.concat(frames, ignore_index=True), PRICE_HISTORY_PATH, ["date", "ticker"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect prices for all historical constituents")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end")
    args = parser.parse_args()
    result = collect_price_history(start=args.start, end=args.end)
    print(f"Saved {len(result):,} rows to {PRICE_HISTORY_PATH}")


if __name__ == "__main__":
    main()
