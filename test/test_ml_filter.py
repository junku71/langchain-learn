import numpy as np
import pandas as pd

import ml.ml_filter as ml_filter
import ml.train_model as train_model
from ml.features import FEATURE_COLUMNS, create_panel_features


def _panel(rows: int = 270, tickers: int = 6):
    dates = pd.bdate_range("2024-01-02", periods=rows)
    price_frames = []
    universe_rows = []
    ticker_values = ["005930.KS"] + [f"{index:06d}.KS" for index in range(1, tickers)]
    for number, ticker in enumerate(ticker_values):
        close = 100 + np.linspace(0, 30 + number * 5, rows) + np.sin(
            np.arange(rows) / (5 + number)
        ) * (2 + number)
        price_frames.append(pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * (100 + number * 10),
        }))
        universe_rows.append({
            "ticker": ticker,
            "market": "KOSPI",
            "sector": "SEMI" if number < 3 else "OTHER",
        })
    economy = pd.DataFrame({
        "KOSPI": np.linspace(2500, 2900, rows),
        "VKOSPI": 18 + np.sin(np.arange(rows) / 20),
        "USD_KRW": np.linspace(1300, 1380, rows),
        "NASDAQ": np.linspace(15000, 19000, rows),
        "SOX": np.linspace(4000, 5500, rows),
    }, index=dates)
    return pd.concat(price_frames), pd.DataFrame(universe_rows), economy


def test_panel_contains_requested_features_and_targets():
    prices, universe, economy = _panel()
    result = create_panel_features(prices, economy, universe)

    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert {"future_return_10D", "future_excess_return_10D", "target"}.issubset(
        result.columns
    )
    assert result.groupby("date")["momentum_rank"].max().dropna().le(1).all()


def test_ensemble_is_saved_and_reused(tmp_path, monkeypatch):
    prices, universe, economy = _panel()
    path = tmp_path / "ensemble.joblib"
    monkeypatch.setattr(train_model, "MODEL_PATH", path)
    monkeypatch.setattr(ml_filter, "MODEL_PATH", path)
    monkeypatch.setattr(train_model, "MODEL_REPORT_PATH", tmp_path / "model_report.json")
    monkeypatch.setattr(train_model, "FEATURE_IMPORTANCE_PATH", tmp_path / "importance.csv")
    monkeypatch.setattr(train_model, "OOS_PREDICTIONS_PATH", tmp_path / "oos.parquet")
    monkeypatch.setattr(train_model, "TOP_SELECTION_PATH", tmp_path / "top.csv")
    monkeypatch.setattr(train_model, "WALK_FORWARD_STEP", 126)

    artifact = train_model.train_market_ensemble(prices, universe, economy)
    market_df = prices[prices["ticker"].eq("005930.KS")].set_index("date")
    result = ml_filter.predict_up_probability("005930.KS", market_df)

    assert path.exists()
    assert set(artifact["models"]) == {
        "lgbm_classifier", "lgbm_ranker", "random_forest"
    }
    assert result["model_reused"] is True
    assert result["algorithm"].startswith("LGBMClassifier+")
    expected = (
        0.3 * result["lgbm_probability"]
        + 0.5 * result["return_rank"]
        + 0.2 * result["rf_probability"]
    )
    assert result["ml_score"] == pytest.approx(expected)


import pytest
