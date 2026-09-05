from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.economy_data import preprocess_economy_data
from analysis.technical import calculate_bollinger_bands, calculate_indicators
from ml.config import EXCESS_RETURN_THRESHOLD, PREDICTION_HORIZON

MOMENTUM_FEATURES = ["ret_5", "ret_20", "ret_60", "sector_relative_momentum"]
TECHNICAL_FEATURES = ["rsi", "adx", "atr_pct", "macd_pct", "bollinger_position", "realized_vol20"]
VOLUME_FEATURES = ["volume_ratio", "turnover", "liquidity"]
FLOW_FEATURES = ["foreign_5_pct", "foreign_20_pct", "institution_5_pct", "institution_20_pct"]
FUNDAMENTAL_FEATURES = [
    "roe", "pbr", "pbr_sector_rank", "per", "per_sector_rank",
    "ev_ebitda_rank", "earnings_growth",
]
MARKET_FEATURES = ["kospi_trend", "vkospi", "usdkrw", "nasdaq", "sp500", "sox", "gold", "oil"]
CROSS_SECTION_FEATURES = ["momentum_rank", "flow_rank", "value_rank"]
FEATURE_COLUMNS = [
    *MOMENTUM_FEATURES, *TECHNICAL_FEATURES, *VOLUME_FEATURES, *FLOW_FEATURES,
    *FUNDAMENTAL_FEATURES, *MARKET_FEATURES, *CROSS_SECTION_FEATURES,
]
TARGET_COLUMNS = ["future_return_10D", "future_excess_return_10D", "target", "future_return_3M", "future_excess_return_3M", "target_3M"]


def _series(frame: pd.DataFrame, name: str | None) -> pd.Series:
    if not name or name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").set_index("date").copy()
    frame = calculate_bollinger_bands(calculate_indicators(frame))
    close = _series(frame, "Close")
    returns = close.pct_change(fill_method=None)
    for days in (5, 20, 60):
        frame[f"ret_{days}"] = close.pct_change(days, fill_method=None)
    frame["rsi"] = _series(frame, "RSI") / 100
    frame["adx"] = _series(frame, "ADX") / 100
    frame["atr_pct"] = _series(frame, "ATR") / close
    frame["macd_pct"] = _series(frame, "MACD") / close
    frame["bollinger_position"] = _series(frame, "BOLLINGER_POSITION")
    frame["realized_vol20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    frame["volume_ratio"] = _series(frame, "VOLUME_RATIO")
    traded_value = close * _series(frame, "Volume")
    market_cap = _series(frame, "market_cap")
    if market_cap.isna().all():
        market_cap = close * _series(frame, "shares_outstanding")
    frame["turnover"] = traded_value / market_cap
    frame["liquidity"] = np.log1p(traded_value.rolling(20, min_periods=20).mean())
    for owner in ("foreign", "institution"):
        net_value = _series(frame, f"{owner}_net") * close
        for days in (5, 20):
            frame[f"{owner}_{days}_pct"] = net_value.rolling(days, min_periods=days).sum() / market_cap

    aliases = {
        "per": ("per", "PER"), "pbr": ("pbr", "PBR"), "roe": ("roe", "ROE"),
        "ev_ebitda": ("ev_ebitda", "EV_EBITDA"),
        "earnings_growth": ("earnings_growth", "operating_profit_growth"),
    }
    for target, candidates in aliases.items():
        source = next((name for name in candidates if name in frame), None)
        frame[target] = _series(frame, source)
    return frame.reset_index()


def _attach_universe(panel: pd.DataFrame, universe: pd.DataFrame | None) -> pd.DataFrame:
    if universe is None or universe.empty:
        return panel
    columns = [column for column in (
        "date", "ticker", "name", "market", "sector", "market_cap", "market_cap_rank",
        "training_universe", "prediction_universe",
    ) if column in universe]
    metadata = universe[columns].copy()
    keys = ["date", "ticker"] if "date" in metadata else ["ticker"]
    metadata = metadata.drop_duplicates(keys, keep="last")
    if "date" in metadata:
        metadata["date"] = pd.to_datetime(metadata["date"]).astype("datetime64[ns]")
        panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    overlap = [column for column in metadata if column not in keys and column in panel]
    return panel.drop(columns=overlap).merge(metadata, on=keys, how="left")


def _winsorize_by_date(panel: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(panel[column], errors="coerce")
        lower = values.groupby(panel["date"]).transform(lambda x: x.quantile(0.01))
        upper = values.groupby(panel["date"]).transform(lambda x: x.quantile(0.99))
        panel[column] = values.clip(lower=lower, upper=upper)


def create_panel_features(
    price_panel: pd.DataFrame,
    economy_df: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    required = {"date", "ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(price_panel)
    if missing:
        raise ValueError(f"Price panel missing columns: {sorted(missing)}")
    raw = _attach_universe(price_panel.copy(), universe)
    panel = pd.concat([_ticker_features(group) for _, group in raw.groupby("ticker", sort=False)], ignore_index=True)
    # Rolling indicators and labels need continuous price history around entry
    # and exit dates, so filter the daily universe only after computing them.
    panel["future_return_10D"] = panel.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )
    panel["future_return_3M"] = panel.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-63) / x - 1
    )
    if "training_universe" in panel:
        panel = panel[panel["training_universe"].fillna(False)].copy()
    panel["sector"] = panel.get("sector", pd.Series(index=panel.index, dtype=object)).replace("UNKNOWN", np.nan)

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    market = preprocess_economy_data(economy_df, dates)
    def level(name: str, lag: int = 0) -> pd.Series:
        values = market[name] if name in market else pd.Series(index=dates, dtype=float)
        return values.shift(lag)
    market_features = pd.DataFrame(index=dates)
    market_features["kospi_trend"] = level("KOSPI").pct_change(20, fill_method=None)
    market_features["vkospi"] = level("VKOSPI")
    market_features["usdkrw"] = level("USD_KRW", lag=1).pct_change(5, fill_method=None)
    for target, source in (("nasdaq", "NASDAQ"), ("sp500", "SP500"), ("sox", "SOX"), ("gold", "GOLD"), ("oil", "WTI")):
        market_features[target] = level(source, lag=1).pct_change(5, fill_method=None)
    market_features["kospi_future_return"] = level("KOSPI").shift(-horizon) / level("KOSPI") - 1
    market_features["kosdaq_future_return"] = level("KOSDAQ").shift(-horizon) / level("KOSDAQ") - 1
    panel = panel.merge(market_features.rename_axis("date").reset_index(), on="date", how="left")

    market_name = panel["market"] if "market" in panel else pd.Series("KOSPI", index=panel.index)
    benchmark = panel["kospi_future_return"].where(market_name.ne("KOSDAQ"), panel["kosdaq_future_return"])

    panel["future_excess_return_10D"] = panel["future_return_10D"] - benchmark
    panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark

    panel["target"] = (panel["future_excess_return_10D"] > EXCESS_RETURN_THRESHOLD).astype("Int64")
    panel["target_3M"] = (panel["future_excess_return_3M"] > 0.0).astype("Int64")

    panel.loc[panel["future_excess_return_10D"].isna(), "target"] = pd.NA
    panel.loc[panel["future_excess_return_3M"].isna(), "target_3M"] = pd.NA

    sector_key = panel.groupby(["date", "sector"], dropna=True)["ret_20"].mean().rename("sector_ret20").reset_index()
    panel = panel.merge(sector_key, on=["date", "sector"], how="left")
    panel["sector_relative_momentum"] = panel["ret_20"] - panel["sector_ret20"]
    for column in ("per", "pbr", "ev_ebitda"):
        panel.loc[panel[column].le(0), column] = np.nan
    by_date = panel.groupby("date")
    panel["pbr_sector_rank"] = panel.groupby(["date", "sector"])["pbr"].rank(pct=True, ascending=False)
    panel["per_sector_rank"] = panel.groupby(["date", "sector"])["per"].rank(pct=True, ascending=False)
    panel["ev_ebitda_rank"] = by_date["ev_ebitda"].rank(pct=True, ascending=False)
    panel["momentum_rank"] = by_date["ret_20"].rank(pct=True)
    flow = panel[["foreign_20_pct", "institution_20_pct"]].mean(axis=1)
    panel["flow_rank"] = flow.groupby(panel["date"]).rank(pct=True)
    value = panel[["pbr_sector_rank", "per_sector_rank", "ev_ebitda_rank"]].mean(axis=1)
    panel["value_rank"] = value.groupby(panel["date"]).rank(pct=True)
    panel["rank_target"] = by_date["future_excess_return_10D"].rank(pct=True).mul(9).round().astype("Int64")
    _winsorize_by_date(panel, [
        "ret_5", "ret_20", "ret_60", "atr_pct", "realized_vol20", "turnover",
        "foreign_5_pct", "foreign_20_pct", "institution_5_pct", "institution_20_pct",
        "roe", "earnings_growth",
    ])
    panel[FEATURE_COLUMNS] = panel[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
