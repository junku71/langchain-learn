from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.economy_data import update_economy_data
from ml.config import (
    FEATURE_PANEL_PATH,
    FINAL_DATASET_START,
    PRICE_HISTORY_PATH,
    UNIVERSE_HISTORY_PATH,
)
from ml.features import FEATURE_COLUMNS, TARGET_COLUMNS, create_panel_features
from ml.panel_data import FLOW_HISTORY_PATH, FUNDAMENTAL_HISTORY_PATH, load_universe
from ml.storage import read_frame, write_frame
from ml.sector_data import update_universe_sectors


def _attach_point_in_time(prices: pd.DataFrame) -> pd.DataFrame:
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    if FLOW_HISTORY_PATH.exists():
        flow = pd.read_csv(FLOW_HISTORY_PATH, parse_dates=["date"], dtype={"ticker": str})
        flow["date"] = pd.to_datetime(flow["date"]).astype("datetime64[ns]")
        panel = panel.merge(
            flow[["date", "ticker", "foreign_net", "institution_net"]],
            on=["date", "ticker"], how="left",
        )
    if FUNDAMENTAL_HISTORY_PATH.exists():
        fundamentals = pd.read_csv(
            FUNDAMENTAL_HISTORY_PATH, parse_dates=["available_date"], dtype={"ticker": str}
        ).sort_values("available_date")
        fundamentals["available_date"] = (
           pd.to_datetime(fundamentals["available_date"]).astype("datetime64[ns]")
        )
        columns = [column for column in (
            "available_date", "per", "pbr", "psr", "pcr", "ev_ebitda", "roe", "earnings_growth",
        ) if column in fundamentals]
        merged = []
        for ticker, group in panel.groupby("ticker", sort=False):
            history = fundamentals[fundamentals["ticker"].eq(ticker)]
            if history.empty:
                merged.append(group)
            else:
                ticker_result = pd.merge_asof(
                    group.sort_values("date"), history[columns].sort_values("available_date"),
                    left_on="date", right_on="available_date", direction="backward",
                ).rename(columns={"available_date": "fundamental_available_date"})
                ticker_result["fundamental_age_days"] = (
                    ticker_result["date"] - ticker_result["fundamental_available_date"]
                ).dt.days
                ticker_result["fundamental_stale"] = ticker_result["fundamental_age_days"].gt(550)
                stale = ticker_result["fundamental_stale"].fillna(False)
                fundamental_values = [column for column in (
                    "per", "pbr", "psr", "pcr", "ev_ebitda", "roe", "earnings_growth",
                ) if column in ticker_result]
                ticker_result.loc[stale, fundamental_values] = pd.NA
                merged.append(ticker_result)
        panel = pd.concat(merged, ignore_index=True)
    return panel


def _filter_final_period(features: pd.DataFrame) -> pd.DataFrame:
    """Trim output only after all lookback-dependent features are calculated."""
    dates = pd.to_datetime(features["date"]).astype("datetime64[ns]")
    return features.loc[dates.ge(pd.Timestamp(FINAL_DATASET_START))].reset_index(drop=True)


def build_dataset(output_path: Path = FEATURE_PANEL_PATH) -> pd.DataFrame:
    universe = read_frame(UNIVERSE_HISTORY_PATH)
    prices = read_frame(PRICE_HISTORY_PATH)
    if universe.empty or prices.empty:
        raise ValueError("Point-in-time universe and price history must be collected first")
    universe["date"] = pd.to_datetime(universe["date"]).astype("datetime64[ns]")
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")

    # 현재 유니버스에 저장된 사용자 지정 ml_sector를 보존한다.
    current_sectors = load_universe(update_sectors=False)
    
    sector_columns = [
        column
        for column in ("ticker", "kis_sector", "kis_market", "ml_sector")
        if column in current_sectors
    ]
    
    # 과거 유니버스에 한 번이라도 등장한 모든 종목
    historical_tickers = universe[["ticker"]].drop_duplicates()
    
    sector_universe = historical_tickers.merge(
        current_sectors[sector_columns].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )
    
    # 캐시에 없는 종목은 KIS API에서 조회한다.
    sectors = update_universe_sectors(sector_universe)
    sector_column = "sector" if "sector" in sectors else "ml_sector"
    sector_map = sectors.set_index("ticker")[sector_column].to_dict()
    universe["sector"] = universe.get("sector", universe["ticker"].map(sector_map))
    universe["sector"] = universe["sector"].replace("UNKNOWN", pd.NA).fillna(
        universe["ticker"].map(sector_map)
    )
    panel = _attach_point_in_time(prices)
    economy = update_economy_data(panel["date"].min(), panel["date"].max())
    features = create_panel_features(panel, economy, universe)
    features = _filter_final_period(features)
    write_frame(features, output_path)
    csv_columns = [column for column in (
        "date", "ticker", "market", "sector", "market_cap", "market_cap_rank",
        "training_universe", "prediction_universe", *FEATURE_COLUMNS, *TARGET_COLUMNS,
    ) if column in features]
    features[csv_columns].to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time ML feature panel")
    parser.parse_args()
    result = build_dataset()
    print(f"Built {len(result):,} rows and {len(FEATURE_COLUMNS)} features at {FEATURE_PANEL_PATH}")


if __name__ == "__main__":
    main()
