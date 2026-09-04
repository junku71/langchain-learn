import pandas as pd

from ml.build_dataset import _filter_final_period
from ml.features import FEATURE_COLUMNS, create_panel_features
from ml.storage import read_frame, write_frame
from ml.validate_dataset import validate_dataset
from ml.check_krx_api import check_krx_api
from ml.universe_history import KrxUniverseProvider, collect_universe_history
import ml.universe_history as universe_history


def test_final_dataset_starts_on_2021_10_01():
    features = pd.DataFrame({
        "date": pd.to_datetime(["2021-09-30", "2021-10-01", "2021-10-04"]),
        "ticker": ["A", "A", "A"],
    })

    result = _filter_final_period(features)

    assert result["date"].min() == pd.Timestamp("2021-10-01")
    assert result["ticker"].tolist() == ["A", "A"]


def _prices():
    dates = pd.bdate_range("2025-01-02", periods=90)
    rows = []
    for ticker, offset in (("000001.KS", 0), ("000002.KS", 10)):
        close = pd.Series(range(100 + offset, 190 + offset), dtype=float)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "Open": close - 1, "High": close + 1, "Low": close - 2,
            "Close": close, "Volume": 1_000_000,
            "foreign_net": 1000, "institution_net": 500,
        }))
    return pd.concat(rows, ignore_index=True)


def test_point_in_time_universe_filters_non_members_and_normalizes_flow():
    prices = _prices()
    dates = prices["date"].unique()
    universe = pd.DataFrame({
        "date": dates, "ticker": "000001.KS", "market": "KOSPI",
        "sector": "SEMI", "market_cap": 10_000_000_000,
        "market_cap_rank": 1, "training_universe": True,
        "prediction_universe": True,
    })
    economy = pd.DataFrame({
        "KOSPI": range(2500, 2590), "KOSDAQ": range(800, 890),
        "NASDAQ": range(15000, 15090), "SP500": range(5000, 5090),
        "SOX": range(4000, 4090), "GOLD": range(2000, 2090),
        "WTI": range(70, 160), "USD_KRW": range(1300, 1390),
        "VKOSPI": 20,
    }, index=pd.DatetimeIndex(dates))
    result = create_panel_features(prices, economy, universe)

    assert set(result["ticker"]) == {"000001.KS"}
    assert set(FEATURE_COLUMNS).issubset(result)
    assert result["foreign_20_pct"].dropna().gt(0).all()


def test_parquet_storage_round_trip(tmp_path):
    path = tmp_path / "frame.parquet"
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "ticker": ["005930.KS"]})
    write_frame(frame, path)
    pd.testing.assert_frame_equal(read_frame(path), frame)


def test_universe_collector_marks_top_100_per_market(tmp_path, monkeypatch):
    class Provider:
        def market_cap_snapshot(self, session):
            rows = []
            for market, suffix, offset in (("KOSPI", "KS", 0), ("KOSDAQ", "KQ", 500)):
                rows.append(pd.DataFrame({
                    "date": pd.Timestamp(session),
                    "ticker": [f"{offset + i:06d}.{suffix}" for i in range(120)],
                    "name": [str(offset + i) for i in range(120)],
                    "market": market,
                    "market_cap": list(range(120, 0, -1)),
                }))
            return pd.concat(rows, ignore_index=True)

    path = tmp_path / "universe.parquet"
    monkeypatch.setattr(universe_history, "UNIVERSE_HISTORY_PATH", path)
    result = collect_universe_history("2025-01-02", "2025-01-02", Provider())
    assert len(result) == 200
    assert result["training_universe"].all()
    assert result["prediction_universe"].sum() == 100
    assert result.groupby("market")["ticker"].size().to_dict() == {
        "KOSDAQ": 100,
        "KOSPI": 100,
    }
    assert result.groupby("market")["market_cap_rank"].max().eq(100).all()


def test_krx_open_api_provider_combines_kospi_and_kosdaq():
    class Response:
        status_code = 200
        def __init__(self, row): self.row = row
        def raise_for_status(self): return None
        def json(self): return {"OutBlock_1": [self.row]}

    class Session:
        def get(self, url, **kwargs):
            market = "KOSDAQ" if "ksq_" in url else "KOSPI"
            code = "000002" if market == "KOSDAQ" else "000001"
            return Response({
                "ISU_CD": code, "ISU_NM": code, "MKT_NM": market,
                "MKTCAP": "1,000,000",
            })

    result = KrxUniverseProvider("key", Session()).market_cap_snapshot(
        pd.Timestamp("2025-01-02").date()
    )
    assert set(result["ticker"]) == {"000001.KS", "000002.KQ"}


def test_quality_report_blocks_training_when_source_groups_are_empty():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02"]),
        "ticker": ["005930.KS"],
        "market_cap_rank": [1],
        **{column: [float("nan")] for column in FEATURE_COLUMNS},
    })
    report = validate_dataset(
        frame, fail=False, require_training_ready=True, write_report=False
    )
    assert report["training_ready"] is False
    assert "insufficient source-group coverage" in report["errors"]


def test_krx_diagnostic_does_not_expose_key(monkeypatch):
    class Response:
        status_code = 401
        def json(self):
            return {"respCode": "401", "respMsg": "Unauthorized API Call"}

    class Session:
        def get(self, url, **kwargs):
            assert kwargs["headers"]["AUTH_KEY"] == "secret"
            return Response()

    monkeypatch.setenv("KRX_API_KEY", "secret")
    results = check_krx_api(session=Session())
    assert len(results) == 4
    assert all(result["http_status"] == 401 for result in results)
    assert "secret" not in str(results)
