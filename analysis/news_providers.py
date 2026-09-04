from typing import Protocol

from analysis.news_models import EarningsEvent, NewsItem


class NewsProvider(Protocol):
    def search(self, ticker: str, count: int = 15) -> list[NewsItem]: ...


class EarningsNewsProvider(NewsProvider, Protocol):
    def search_earnings_news(
        self,
        ticker: str,
        count: int = 10,
    ) -> list[NewsItem]: ...


class EarningsProvider(Protocol):
    def get_consensus(self, ticker: str) -> EarningsEvent: ...
