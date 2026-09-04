from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol

import joblib
import pandas as pd

from broker.base import Broker
from broker.kis import KISBroker
from ml.config import FEATURE_PANEL_PATH
from ml.storage import read_frame
from ml.train_model import MODEL_PATH, MODEL_VERSION
from trading.models import MinuteBar


class QuoteProvider(Protocol):
    def minute_bar(self, ticker: str, now: datetime) -> MinuteBar: ...


class CandidateProvider(Protocol):
    def candidates(self, trade_date: date, per_market: int) -> list[dict]: ...


class CandidateAnalysis(Protocol):
    def evaluate(self, candidate: dict, bar: MinuteBar) -> dict: ...


class BrokerQuoteProvider:
    """Point-in-time quote adapter.

    Until a true minute OHLC endpoint is connected, O/H/L/C are the same live
    quote. This safely detects point-price stops but cannot reconstruct an
    intraminute high/low that occurred between scheduler ticks.
    """

    def __init__(self, broker: Broker):
        self.broker = broker

    def minute_bar(self, ticker: str, now: datetime) -> MinuteBar:
        price = float(self.broker.get_current_price(ticker))
        return MinuteBar(
            ticker, now, price, price, price, price,
            source="CURRENT_PRICE_FALLBACK",
        )


class KISMinuteBarProvider:
    """KIS completed-minute OHLC provider with an optional live-price fallback."""

    def __init__(
        self,
        broker: KISBroker,
        *,
        fallback_to_current_price: bool = True,
        completed_bars_only: bool = True,
    ):
        self.broker = broker
        self.completed_bars_only = completed_bars_only
        self.fallback = (
            BrokerQuoteProvider(broker) if fallback_to_current_price else None
        )
        self._cache: dict[tuple[str, datetime], MinuteBar] = {}

    def minute_bar(self, ticker: str, now: datetime) -> MinuteBar:
        minute = now.replace(second=0, microsecond=0)
        key = (ticker, minute)
        if key in self._cache:
            return self._cache[key]

        try:
            bars = self.broker.get_minute_bars(ticker, now)
            cutoff = minute - timedelta(minutes=1)
            eligible = [
                row for row in bars
                if row["timestamp"] <= (
                    cutoff if self.completed_bars_only else now
                )
            ]
            if not eligible:
                raise ValueError(f"No eligible KIS minute bar for {ticker}")
            row = eligible[-1]
            bar = MinuteBar(
                ticker=ticker,
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                source="KIS_MINUTE",
            )
        except Exception:
            if self.fallback is None:
                raise
            bar = self.fallback.minute_bar(ticker, now)

        # Keep only the current scheduler bucket; this also bounds long-running
        # scheduler memory without a separate cache dependency.
        self._cache = {
            cached_key: cached_bar
            for cached_key, cached_bar in self._cache.items()
            if cached_key[1] == minute
        }
        self._cache[key] = bar
        return bar


class PassthroughCandidateAnalysis:
    def evaluate(self, candidate: dict, bar: MinuteBar) -> dict:
        return {
            "approved": True,
            "ticker": candidate["ticker"],
            "price": bar.close,
            "classification_probability": float(
                candidate["classification_probability"]
            ),
            "ml_rank": int(candidate["ml_rank"]),
            "reason": "PRECOMPUTED_MODEL_CANDIDATE",
        }


class ModelCandidateProvider:
    def __init__(self, model_path=MODEL_PATH, panel_path=FEATURE_PANEL_PATH):
        self.model_path = model_path
        self.panel_path = panel_path

    def candidates(self, trade_date: date, per_market: int) -> list[dict]:
        if not self.model_path.exists():
            raise ValueError(f"Trained model not found: {self.model_path}")
        artifact = joblib.load(self.model_path)
        metadata = artifact.get("metadata", {})
        if metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("Model version is stale; retrain before live trading")
        trained_through = pd.Timestamp(metadata["trained_through"]).date()
        if trained_through >= trade_date:
            raise ValueError("Model trained_through must precede the live trade date")

        predictions = pd.DataFrame.from_dict(
            artifact.get("latest_predictions", {}), orient="index"
        )
        if predictions.empty:
            return []
        predictions.index.name = "ticker"
        predictions = predictions.reset_index()

        panel = read_frame(self.panel_path)
        latest = pd.DataFrame()
        if not panel.empty:
            latest_date = pd.to_datetime(panel["date"]).max()
            columns = [
                column for column in ("ticker", "sector", "atr_pct")
                if column in panel
            ]
            latest = panel.loc[pd.to_datetime(panel["date"]).eq(latest_date), columns]
            latest = latest.drop_duplicates("ticker", keep="last")
        if not latest.empty:
            predictions = predictions.merge(latest, on="ticker", how="left", suffixes=("", "_panel"))
            if "sector_panel" in predictions:
                if "sector" in predictions:
                    predictions["sector"] = predictions["sector"].fillna(
                        predictions["sector_panel"]
                    )
                else:
                    predictions["sector"] = predictions["sector_panel"]

        required = {"ticker", "market", "ml_score", "classification_probability", "ml_rank"}
        missing = required - set(predictions)
        if missing:
            raise ValueError(f"Model predictions missing columns: {sorted(missing)}")
        ranked = predictions.sort_values(
            ["market", "ml_score"], ascending=[True, False]
        ).groupby("market", group_keys=False).head(per_market)
        ranked["sector"] = ranked.get(
            "sector", pd.Series(index=ranked.index, dtype=object)
        ).fillna("UNKNOWN")
        return ranked.to_dict("records")
