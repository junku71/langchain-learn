import numpy as np
import pandas as pd
import pytest

from analysis import technical


@pytest.fixture
def market_data() -> pd.DataFrame:
    periods = 120
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    steps = np.arange(periods, dtype=float)
    close = 100 + (steps * 0.35) + (np.sin(steps / 3) * 2)

    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 1.2,
            "Low": close - 1.1,
            "Close": close,
            "Volume": 1_000_000 + (steps * 2_000),
        },
        index=index,
    )


def test_get_stock_data_normalizes_yfinance_multiindex(monkeypatch):
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["005930.KS"]]
    )
    raw = pd.DataFrame(
        [
            [100, 102, 99, 101, 1000],
            [np.nan, np.nan, np.nan, np.nan, 1200],
        ],
        columns=columns,
    )
    monkeypatch.setattr(technical.yf, "download", lambda *args, **kwargs: raw)

    result = technical.get_stock_data("005930.KS")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(result) == 1
    assert result.iloc[-1]["Close"] == 101


def test_calculate_indicators_creates_complete_latest_row(market_data):
    result = technical.calculate_indicators(market_data)
    expected = {
        "MA5",
        "MA20",
        "MA60",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "ATR",
        "VOLUME_RATIO",
        "ADX",
        "DI_PLUS",
        "DI_MINUS",
    }

    assert expected.issubset(result.columns)
    assert np.isfinite(result.iloc[-1][list(expected)].astype(float)).all()
    assert set(market_data.columns) == {"Open", "High", "Low", "Close", "Volume"}


@pytest.mark.parametrize(
    ("score", "signal"),
    [(0, "SELL"), (30, "SELL"), (31, "HOLD"), (69, "HOLD"), (70, "BUY")],
)
def test_make_trading_signal_boundaries(score, signal):
    assert technical.make_trading_signal(score) == signal


def test_get_technical_analysis_uses_market_data(monkeypatch, market_data):
    monkeypatch.setattr(
        technical,
        "get_stock_data",
        lambda ticker: market_data.copy(),
    )

    result = technical.get_technical_analysis("005930.KS")

    assert result["ticker"] == "005930.KS"
    assert result["price"] == pytest.approx(market_data.iloc[-1]["Close"])
    assert 0 <= result["score"] <= 100
    assert result["signal"] in {"BUY", "HOLD", "SELL"}
    assert set(result["analysis"]) == {
        "trend",
        "rsi_signal",
        "macd_signal",
        "volume_signal",
        "trend_strength",
        "di_signal",
    }
    assert np.isfinite(list(result["indicators"].values())).all()


def test_get_technical_analysis_reuses_supplied_snapshot(monkeypatch, market_data):
    monkeypatch.setattr(
        technical, "get_stock_data",
        lambda ticker: pytest.fail("shared snapshot should avoid another download"),
    )

    result = technical.get_technical_analysis("005930.KS", market_data)

    metrics = result["derived_metrics"]
    assert result["data_points"] == len(market_data)
    assert metrics["return_5d"] is not None
    assert metrics["return_20d"] is not None
    assert metrics["atr_pct"] > 0
    assert isinstance(metrics["breakout_20d"], bool)


def test_get_stock_price_uses_latest_row(monkeypatch, market_data):
    monkeypatch.setattr(
        technical,
        "get_stock_data",
        lambda ticker: market_data.copy(),
    )

    result = technical.get_stock_price("005930.KS")
    latest = market_data.iloc[-1]

    assert result == {
        "ticker": "005930.KS",
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"]),
    }


def test_calculate_risk_uses_atr_position_sizing(monkeypatch, market_data):
    monkeypatch.setattr(
        technical,
        "get_stock_data",
        lambda ticker: market_data.copy(),
    )

    result = technical.calculate_risk(
        "005930.KS",
        account_size=10_000_000,
        risk_per_trade=0.01,
    )

    assert result["risk_amount"] == 100_000
    assert result["stop_loss"] == pytest.approx(
        result["price"] - (2 * result["ATR"])
    )
    assert result["take_profit"] == pytest.approx(
        result["price"] + (3 * result["ATR"])
    )
    assert result["position_size"] == int(
        result["risk_amount"] / (2 * result["ATR"])
    )
