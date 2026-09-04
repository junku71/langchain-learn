from pathlib import Path

import pandas as pd

from analysis.economy_data import (
    ECONOMIC_FEATURE_COLUMNS,
    KrxVkospiProvider,
    add_economic_features,
    preprocess_economy_data,
    update_economy_data,
)


class FakeKrxResponse:
    status_code = 200

    def __init__(self, requested_date):
        self.requested_date = requested_date

    def raise_for_status(self):
        return None

    def json(self):
        if self.requested_date == "20250102":
            return {
                "OutBlock_1": [
                    {"IDX_NM": "코스피 200", "CLSPRC_IDX": "320.10"},
                    {"IDX_NM": "코스피 200 변동성지수", "CLSPRC_IDX": "18.75"},
                ]
            }
        return {"OutBlock_1": []}


class FakeKrxSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return FakeKrxResponse(params["basDd"])


class FakeProvider:
    def __init__(self):
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        index = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"SP500": range(100, 100 + len(index))}, index=index)


def test_cache_updates_only_missing_tail_and_deduplicates(tmp_path: Path):
    provider = FakeProvider()
    path = tmp_path / "economy.csv"

    update_economy_data("2025-01-01", "2025-01-03", path, [provider])
    result = update_economy_data("2025-01-01", "2025-01-05", path, [provider])

    assert provider.calls[-1][0].isoformat() == "2025-01-04"
    assert result.index.is_unique
    assert len(result) == 5


def test_vkospi_uses_krx_and_skips_checkpointed_dates(tmp_path: Path):
    session = FakeKrxSession()
    provider = KrxVkospiProvider(
        "test-key",
        cache_path=tmp_path / "vkospi.csv",
        session=session,
        checkpoint_every=1,
        request_interval=0,
    )

    first = provider.fetch(pd.Timestamp("2025-01-02").date(), pd.Timestamp("2025-01-03").date())
    second = provider.fetch(pd.Timestamp("2025-01-02").date(), pd.Timestamp("2025-01-03").date())

    assert first.loc["2025-01-02", "VKOSPI"] == 18.75
    assert pd.isna(first.loc["2025-01-03", "VKOSPI"])
    assert second.equals(first)
    assert len(session.calls) == 2
    assert session.calls[0][2] == {"AUTH_KEY": "test-key"}


def test_features_align_to_stock_days_without_lookahead():
    stock_index = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    stock = pd.DataFrame({"Close": [100, 101, 102]}, index=stock_index)
    economy = pd.DataFrame(
        {"SP500": [100, 110, 121]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    result = add_economic_features(stock, economy, update=False)

    assert round(result.loc["2025-01-03", "ECON_SP500_RETURN_1D"], 4) == 0.1
    assert set(ECONOMIC_FEATURE_COLUMNS).issubset(result.columns)


def test_forward_fill_uses_only_past_observations():
    economy = pd.DataFrame(
        {"SP500": [100.0, 110.0]},
        index=pd.to_datetime(["2025-01-03", "2025-01-07"]),
    )
    sessions = pd.to_datetime(["2025-01-02", "2025-01-06", "2025-01-07"])

    result = preprocess_economy_data(economy, sessions)

    assert pd.isna(result.loc["2025-01-02", "SP500"])
    assert result.loc["2025-01-06", "SP500"] == 100.0
    assert result.loc["2025-01-07", "SP500"] == 110.0
