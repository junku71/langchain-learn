import html
import os
import re

import requests

from analysis.news_models import NewsItem
from analysis.ticker_mapper import get_company_name


class NaverNewsProvider:
    DEVELOPERS_URL = "https://openapi.naver.com/v1/search/news.json"
    API_HUB_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        api_hub_key_id: str | None = None,
        api_hub_key: str | None = None,
        url: str | None = None,
        session: requests.Session | None = None,
    ):
        hub_key_id = api_hub_key_id or os.getenv(
            "X-NCP-APIGW-API-KEY-ID", ""
        )
        hub_key = api_hub_key or os.getenv("X-NCP-APIGW-API-KEY", "")
        developers_id = client_id or os.getenv("NAVER_CLIENT_ID", "")
        developers_secret = client_secret or os.getenv(
            "NAVER_CLIENT_SECRET", ""
        )

        if hub_key_id and hub_key:
            self.credentials = {
                "X-NCP-APIGW-API-KEY-ID": hub_key_id,
                "X-NCP-APIGW-API-KEY": hub_key,
            }
            self.url = url or self.API_HUB_URL
            self.provider_name = "NAVER_API_HUB"
        else:
            self.credentials = {
                "X-Naver-Client-Id": developers_id,
                "X-Naver-Client-Secret": developers_secret,
            }
            self.url = url or self.DEVELOPERS_URL
            self.provider_name = "NAVER_DEVELOPERS"

        self.session = session or requests.Session()

    @staticmethod
    def _clean(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"<[^>]+>", "", html.unescape(text)).strip()

    def _search(self, query: str, display: int, source: str) -> list[NewsItem]:
        if not all(self.credentials.values()):
            return []

        response = self.session.get(
            self.url,
            headers=self.credentials,
            params={"query": query, "display": display, "sort": "date"},
            timeout=10,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        return [
            NewsItem(
                source=source,
                title=self._clean(item.get("title")),
                link=item.get("originallink") or item.get("link"),
                published_at=item.get("pubDate"),
                description=self._clean(item.get("description")),
            )
            for item in response.json().get("items", [])
        ]

    def search(self, ticker: str, display: int = 20) -> list[NewsItem]:
        return self._search(
            get_company_name(ticker),
            display,
            self.provider_name,
        )

    def search_query(self, query: str, display: int = 20) -> list[NewsItem]:
        """Search general market news without treating the query as a ticker."""
        return self._search(query, display, f"{self.provider_name}_MARKET")

    def search_earnings_news(
        self,
        ticker: str,
        display: int = 10,
    ) -> list[NewsItem]:
        company = get_company_name(ticker)
        query = f"{company} 실적 컨센서스 영업이익 매출 전망 실적발표"
        return self._search(
            query,
            display,
            f"{self.provider_name}_EARNINGS",
        )
