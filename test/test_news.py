from analysis.earnings_features import create_earnings_features
from analysis.news_models import EarningsEvent, NewsItem
from analysis.news_naver import NaverNewsProvider
from analysis.news_service import NewsAnalysisService
from dotenv import load_dotenv
import sys


class FakeNaver:
    def search(self, ticker, display):
        return [NewsItem("NAVER", "신규 수주", "https://example.com/1")]

    def search_earnings_news(self, ticker, display):
        return [NewsItem("NAVER_EARNINGS", "실적 전망 상향")]


class FakeYahoo:
    def search(self, ticker, count):
        return [NewsItem("YAHOO", "Global demand improves")]


class FakeEarningsProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def get_consensus(self, ticker):
        if self.error:
            raise self.error
        return self.result


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"items": [{"title": "<b>테스트</b> 뉴스"}]}


class RecordingSession:
    def __init__(self):
        self.request = None

    def get(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        return FakeResponse()


def test_naver_clean_removes_markup():
    assert NaverNewsProvider._clean("<b>삼성전자</b> &amp; 반도체") == (
        "삼성전자 & 반도체"
    )


def test_naver_api_hub_credentials_are_selected():
    session = RecordingSession()
    provider = NaverNewsProvider(
        api_hub_key_id="test-id",
        api_hub_key="test-key",
        session=session,
    )

    result = provider.search("005930.KS", display=1)

    assert session.request["url"] == NaverNewsProvider.API_HUB_URL
    assert session.request["headers"] == {
        "X-NCP-APIGW-API-KEY-ID": "test-id",
        "X-NCP-APIGW-API-KEY": "test-key",
    }
    assert result[0].source == "NAVER_API_HUB"
    assert result[0].title == "테스트 뉴스"


def test_earnings_features():
    result = create_earnings_features({
        "eps_estimate": 1200,
        "eps_30d_ago": 1000,
        "eps_up_30d": 3,
        "eps_down_30d": 1,
        "analyst_target_current": 80000,
        "analyst_target_mean": 100000,
        "days_to_earnings": 5,
    })

    assert result["eps_revision_30d"] == 0.2
    assert result["revision_balance"] == 0.5
    assert result["target_upside"] == 0.25
    assert result["earnings_imminent"] is True


def test_news_service_collects_all_sources():
    earnings = EarningsEvent(eps_estimate=1200, eps_30d_ago=1000)
    service = NewsAnalysisService(
        naver=FakeNaver(),
        yahoo=FakeYahoo(),
        earnings_provider=FakeEarningsProvider(earnings),
    )
    result = service.collect("005930.KS")

    assert len(result["naver_news"]) == 1
    assert len(result["naver_earnings_news"]) == 1
    assert len(result["yahoo_news"]) == 1
    assert result["earnings_features"]["eps_revision_30d"] == 0.2
    assert result["collection_status"]["naver_news"]["status"] == "SUCCESS"


def test_missing_consensus_remains_none():
    features = create_earnings_features({})

    assert features == {
        "eps_revision_30d": None,
        "target_upside": None,
        "revision_balance": None,
        "earnings_imminent": None,
    }


def test_provider_failure_returns_stable_none_schema():
    service = NewsAnalysisService(
        naver=FakeNaver(),
        yahoo=FakeYahoo(),
        earnings_provider=FakeEarningsProvider(error=RuntimeError("offline")),
    )
    result = service.collect("005930.KS")

    assert result["earnings"]["eps_estimate"] is None
    assert result["earnings_features"]["revision_balance"] is None
    assert result["collection_status"]["earnings"]["status"] == "ERROR"


def display_value(value, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if percent and isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def console_text(value) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def print_headlines(label: str, items: list[dict], limit: int = 5) -> None:
    print(f"\n[{label}] {len(items)} articles")

    if not items:
        print("  No data")
        return

    for index, item in enumerate(items[:limit], start=1):
        title = console_text(item.get("title") or "(untitled)")
        source = console_text(item.get("source", "N/A"))
        print(f"  {index}. {title}")
        print(f"     Source: {source}")
        if item.get("link"):
            print(f"     Link: {console_text(item['link'])}")


def print_news_report(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"News Analysis Data: {result['ticker']}")
    print("=" * 60)
    print_headlines("Naver News", result["naver_news"])
    print_headlines("Naver Earnings News", result["naver_earnings_news"])
    print_headlines("Yahoo News", result["yahoo_news"])

    earnings = result["earnings"]
    features = result["earnings_features"]
    print("\n[Earnings / Consensus]")
    fields = [
        ("Next earnings", earnings.get("earnings_date")),
        ("Days to earnings", earnings.get("days_to_earnings")),
        ("EPS estimate", earnings.get("eps_estimate")),
        ("EPS range", f"{display_value(earnings.get('eps_low'))} ~ "
         f"{display_value(earnings.get('eps_high'))}"),
        ("EPS analysts", earnings.get("eps_analysts")),
        ("Revenue estimate", earnings.get("revenue_estimate")),
        ("Revenue analysts", earnings.get("revenue_analysts")),
        ("Target current", earnings.get("analyst_target_current")),
        ("Target mean", earnings.get("analyst_target_mean")),
    ]
    for label, value in fields:
        print(f"  {label:<20}: {display_value(value)}")

    print("\n[Derived Features]")
    print(
        "  EPS revision 30d    : "
        f"{display_value(features.get('eps_revision_30d'), percent=True)}"
    )
    print(
        "  Revision balance   : "
        f"{display_value(features.get('revision_balance'), percent=True)}"
    )
    print(
        "  Target upside      : "
        f"{display_value(features.get('target_upside'), percent=True)}"
    )
    print(
        "  Earnings imminent  : "
        f"{display_value(features.get('earnings_imminent'))}"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tests = [
        test_naver_clean_removes_markup,
        test_naver_api_hub_credentials_are_selected,
        test_earnings_features,
        test_news_service_collects_all_sources,
        test_missing_consensus_remains_none,
        test_provider_failure_returns_stable_none_schema,
    ]

    print("[Local Tests]")
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    load_dotenv()
    result = NewsAnalysisService().collect("005930.KS")
    print_news_report(result)


if __name__ == "__main__":
    main()
