import pandas as pd

from ml.panel_data import attach_point_in_time_data


def test_point_in_time_fundamentals_do_not_backfill_future_release(tmp_path):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    prices = pd.DataFrame({
        "date": dates,
        "ticker": "005930.KS",
        "Close": [100, 101, 102],
    })
    fundamentals = pd.DataFrame({
        "available_date": ["2025-01-03"],
        "ticker": ["005930.KS"],
        "per": [12.0], "pbr": [1.2], "psr": [2.0], "pcr": [8.0],
        "ev_ebitda": [6.0], "roe": [15.0],
    })
    path = tmp_path / "fundamentals.csv"
    fundamentals.to_csv(path, index=False)

    result = attach_point_in_time_data(
        prices,
        flow_path=tmp_path / "missing-flow.csv",
        fundamental_path=path,
    )

    assert pd.isna(result.loc[result["date"].eq(dates[0]), "per"].iloc[0])
    assert result.loc[result["date"].eq(dates[1]), "per"].iloc[0] == 12.0
    assert result.loc[result["date"].eq(dates[2]), "per"].iloc[0] == 12.0
