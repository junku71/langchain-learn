import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.fundamental import analyze_fundamental, calculate_fundamental_score
from broker.kis import KISBroker
from dotenv import load_dotenv


class FakeFundamentalProvider:
    def get_fundamental_data(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "PER": 12.5,
            "PBR": 1.4,
            "ROE": 13.8,
            "debt_ratio": 42.0,
            "PCR": 9.0,
            "EV_EBITDA": 8.5,
            "revenue_growth": 12.0,
            "operating_profit_growth": 18.0,
        }


def test_calculate_fundamental_score():
    result = calculate_fundamental_score(12.5, 1.4, 13.8, 42.0)

    assert result["score"] == 64.2
    assert result["signal"] == "NEUTRAL"
    assert result["confidence"] == "MEDIUM"


def test_new_ratios_are_normalized_and_missing_is_not_zero():
    result = calculate_fundamental_score(
        per=12,
        pbr=1.2,
        roe=16,
        debt_ratio=45,
        pcr=7,
        ev_ebitda=5,
        revenue_growth=22,
        operating_profit_growth=30,
    )

    assert result["score"] == 93.0
    assert result["data_coverage_pct"] == 100.0
    assert result["component_scores"]["PCR"] == 100


def test_missing_metrics_renormalize_available_weights():
    result = calculate_fundamental_score(10, None, None, None)

    assert result["raw_score"] == 100.0
    assert result["score"] == 57.5
    assert result["data_coverage_pct"] == 15.0
    assert result["confidence"] == "LOW"
    assert result["PCR"] is None


def test_analyze_fundamental_uses_provider():
    result = analyze_fundamental("005930.KS", FakeFundamentalProvider())

    assert result["ticker"] == "005930.KS"
    assert result["ROE"] == 13.8


def test_latest_ratio_selects_latest_valid_period():
    rows = [
        {"stac_yymm": "202212", "ratio": "9.5"},
        {"stac_yymm": "202412", "ratio": "-"},
        {"stac_yymm": "202312", "ratio": "11.2"},
    ]

    assert KISBroker._latest_ratio(rows, "ratio") == 11.2


def print_analysis(result: dict) -> None:
    print("\n[KIS Fundamental Analysis]")
    print(f"Ticker     : {result['ticker']}")
    print(f"PER        : {result['PER']}")
    print(f"PBR        : {result['PBR']}")
    print(f"ROE        : {result['ROE']}%")
    print(f"Debt ratio : {result['debt_ratio']}%")
    print(f"PCR        : {result['PCR']}")
    print(f"EV/EBITDA  : {result['EV_EBITDA']}")
    print(f"Sales YoY  : {result['revenue_growth']}%")
    print(f"Op profit  : {result['operating_profit_growth']}%")
    print(f"Coverage   : {result['data_coverage_pct']}%")
    print(f"Score      : {result['score']}/100")
    print(f"Signal     : {result['signal']}")


def main() -> None:
    tests = [
        test_calculate_fundamental_score,
        test_analyze_fundamental_uses_provider,
        test_latest_ratio_selects_latest_valid_period,
    ]

    print("[Local Tests]")

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    load_dotenv()

    try:
        result = analyze_fundamental("005930.KS")
    except Exception as error:
        print(f"\nKIS API test failed: {error}")
        raise SystemExit(1) from error

    print_analysis(result)


if __name__ == "__main__":
    main()
