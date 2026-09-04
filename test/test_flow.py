import pandas as pd

from analysis.flow import (
    analyze_flow,
    calculate_flow_indicators,
    calculate_flow_score,
    standardize_flow_data,
)


class FakeFlowProvider:
    def get_investor_flow(self, ticker: str) -> list[dict]:
        return [
            {
                "stck_bsop_date": f"2026010{day}",
                "frgn_ntby_qty": foreign,
                "orgn_ntby_qty": institution,
            }
            for day, foreign, institution in [
                (1, "100", "50"),
                (2, "120", "40"),
                (3, "80", "-10"),
                (4, "110", "30"),
                (5, "90", "60"),
            ]
        ]


def test_standardize_flow_data_sorts_dates():
    raw = pd.DataFrame(FakeFlowProvider().get_investor_flow("005930.KS"))
    result = standardize_flow_data(raw.iloc[::-1])

    assert result.iloc[0]["date"] < result.iloc[-1]["date"]


def test_flow_indicators_and_score():
    raw = pd.DataFrame(FakeFlowProvider().get_investor_flow("005930.KS"))
    indicators = calculate_flow_indicators(standardize_flow_data(raw))
    result = calculate_flow_score(indicators)

    assert result["joint_buy_days"] == 4
    assert result["foreign_persistence"] == 1.0
    assert result["score"] == 100
    assert result["signal"] == "BULLISH"


def test_analyze_flow_uses_provider():
    result = analyze_flow("005930.KS", provider=FakeFlowProvider())

    assert result["ticker"] == "005930.KS"
    assert result["lookback_days"] == 5
    assert set(result["windows"]) == {"1d", "3d", "5d", "10d", "20d"}
    assert result["windows"]["3d"]["available_days"] == 3
    assert result["price_flow_relationship"] == "UNKNOWN"
