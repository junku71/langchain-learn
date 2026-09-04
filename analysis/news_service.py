from analysis.earnings_features import create_earnings_features
from analysis.earnings_yahoo import YahooEarningsProvider
from analysis.news_models import EarningsEvent
from analysis.news_naver import NaverNewsProvider
from analysis.news_providers import (
    EarningsNewsProvider,
    EarningsProvider,
    NewsProvider,
)
from analysis.news_yahoo import YahooNewsProvider


class NewsAnalysisService:
    def __init__(
        self,
        naver: EarningsNewsProvider | None = None,
        yahoo: NewsProvider | None = None,
        earnings_provider: EarningsProvider | None = None,
    ):
        self.naver = naver or NaverNewsProvider()
        self.yahoo = yahoo or YahooNewsProvider()
        self.earnings_provider = earnings_provider or YahooEarningsProvider()

    @staticmethod
    def _try_collect(callback) -> tuple[list, dict]:
        try:
            items = callback()
            return items, {
                "status": "SUCCESS" if items else "NO_DATA",
                "count": len(items), "error": None,
            }
        except Exception as error:
            return [], {
                "status": "ERROR", "count": 0,
                "error": f"{type(error).__name__}: {error}",
            }

    def collect(self, ticker: str) -> dict:
        naver_news, naver_status = self._try_collect(
            lambda: self.naver.search(ticker, 15)
        )
        earnings_news, earnings_news_status = self._try_collect(
            lambda: self.naver.search_earnings_news(ticker, 10)
        )
        yahoo_news, yahoo_status = self._try_collect(
            lambda: self.yahoo.search(ticker, 15)
        )

        try:
            earnings = self.earnings_provider.get_consensus(ticker).to_dict()
            earnings_status = {
                "status": "SUCCESS" if any(value is not None for value in earnings.values()) else "NO_DATA",
                "error": None,
            }
        except Exception as error:
            earnings = EarningsEvent().to_dict()
            earnings_status = {
                "status": "ERROR",
                "error": f"{type(error).__name__}: {error}",
            }

        return {
            "ticker": ticker,
            "naver_news": [item.to_dict() for item in naver_news],
            "naver_earnings_news": [item.to_dict() for item in earnings_news],
            "yahoo_news": [item.to_dict() for item in yahoo_news],
            "earnings": earnings,
            "earnings_features": create_earnings_features(earnings),
            "collection_status": {
                "naver_news": naver_status,
                "naver_earnings_news": earnings_news_status,
                "yahoo_news": yahoo_status,
                "earnings": earnings_status,
            },
        }
