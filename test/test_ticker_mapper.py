import pandas as pd
import pytest

from analysis.ticker_mapper import TickerMapper


def write_market_csv(directory, market, rows):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        directory / f"{market.lower()}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_reads_existing_csv_without_downloading(tmp_path, monkeypatch):
    write_market_csv(tmp_path, "kospi", [{
        "market": "KOSPI", "code": "005930", "name": "삼성전자",
        "english_name": "Samsung Electronics", "ticker": "005930.KS",
    }])
    mapper = TickerMapper(tmp_path)
    monkeypatch.setattr(
        mapper, "download_market",
        lambda market: (_ for _ in ()).throw(AssertionError("downloaded")),
    )

    assert mapper.code_to_name("005930.KS", "KOSPI") == "삼성전자"
    assert mapper.name_to_code("삼성전자", "KOSPI") == "005930"
    assert mapper.name_to_ticker("삼성전자", "KOSPI") == "005930.KS"


def test_missing_csv_downloads_and_saves(tmp_path, monkeypatch):
    mapper = TickerMapper(tmp_path)

    def fake_download(market):
        frame = pd.DataFrame([{
            "market": market, "code": "AAPL", "name": "애플",
            "english_name": "Apple Inc.", "ticker": "AAPL",
        }])
        mapper.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(mapper.csv_path(market), index=False, encoding="utf-8-sig")
        mapper._frames[market] = frame
        return frame.copy()

    monkeypatch.setattr(mapper, "download_market", fake_download)

    assert mapper.name_to_code("Apple Inc.", "NASDAQ") == "AAPL"
    assert mapper.csv_path("NASDAQ").exists()


def test_market_aliases():
    assert TickerMapper.normalize_market("S&P 500") == "SP500"
    assert TickerMapper.normalize_market("kosdaq") == "KOSDAQ"


def test_sp500_uses_existing_csv(tmp_path):
    write_market_csv(
        tmp_path,
        "sp500",
        [{"종목코드": "MSFT", "종목명": "Microsoft"}],
    )
    mapper = TickerMapper(tmp_path)

    assert mapper.name_to_code("Microsoft", "SP500") == "MSFT"


def test_sp500_missing_membership_is_not_mislabeled(tmp_path, monkeypatch):
    mapper = TickerMapper(tmp_path)
    empty_members = pd.DataFrame({
        "market": ["NASDAQ"], "code": ["AAPL"], "name": ["Apple"],
        "english_name": ["Apple Inc."], "ticker": ["AAPL"],
        "index_member": ["0"],
    })
    monkeypatch.setattr(
        mapper,
        "_download_overseas_exchange",
        lambda exchange: empty_members.copy(),
    )

    with pytest.raises(NotImplementedError, match="S&P 500"):
        mapper.load_market("SP500")
