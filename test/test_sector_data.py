import pandas as pd

from ml.sector_data import update_universe_sectors


class FakeSectorProvider:
    def __init__(self):
        self.calls = []

    def get_stock_sector(self, ticker):
        self.calls.append(ticker)
        return {
            "code": ticker.split(".", 1)[0],
            "kis_sector": "전기전자",
            "kis_market": "KOSPI",
        }


def test_sector_cache_avoids_repeated_api_calls(tmp_path):
    universe = pd.DataFrame({
        "ticker": ["005930.KS"],
        "name": ["삼성전자"],
        "sector": ["UNKNOWN"],
    })
    cache = tmp_path / "sectors.csv"
    first_provider = FakeSectorProvider()
    first = update_universe_sectors(universe, first_provider, cache)
    second_provider = FakeSectorProvider()
    second = update_universe_sectors(universe, second_provider, cache)

    assert first_provider.calls == ["005930.KS"]
    assert second_provider.calls == []
    assert first.loc[0, "kis_sector"] == "전기전자"
    assert second.loc[0, "sector"] == "전기전자"


def test_manual_ml_sector_is_preserved(tmp_path):
    universe = pd.DataFrame({
        "ticker": ["005930.KS"],
        "kis_sector": ["전기전자"],
        "ml_sector": ["Semiconductor"],
    })

    result = update_universe_sectors(
        universe,
        FakeSectorProvider(),
        tmp_path / "sectors.csv",
    )

    assert result.loc[0, "sector"] == "Semiconductor"


def test_foreign_listed_stock_gets_ml_fallback(tmp_path):
    universe = pd.DataFrame({
        "ticker": ["950160.KQ"],
        "kis_sector": ["UNKNOWN"],
        "ml_sector": ["UNKNOWN"],
    })

    class EmptyProvider:
        def get_stock_sector(self, ticker):
            return {"code": "950160", "kis_sector": "UNKNOWN"}

    provider = EmptyProvider()
    result = update_universe_sectors(
        universe,
        provider,
        tmp_path / "sectors.csv",
    )
    second = update_universe_sectors(
        universe,
        provider,
        tmp_path / "sectors.csv",
    )

    assert result.loc[0, "kis_sector"] == "UNKNOWN"
    assert result.loc[0, "ml_sector"] == "FOREIGN_LISTED"
    assert second.loc[0, "ml_sector"] == "FOREIGN_LISTED"
