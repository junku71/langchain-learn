import yfinance as yf

from analysis.news_models import NewsItem


class YahooNewsProvider:
    def search(self, ticker: str, count: int = 15) -> list[NewsItem]:
        raw_news = yf.Ticker(ticker).get_news(count=count, tab="news")
        result = []

        for item in raw_news:
            content = item.get("content", item)
            provider = content.get("provider", {})
            provider_name = (
                provider.get("displayName")
                if isinstance(provider, dict)
                else None
            )
            canonical = content.get("canonicalUrl", {})
            link = (
                canonical.get("url")
                if isinstance(canonical, dict)
                else None
            ) or item.get("link")
            result.append(NewsItem(
                source=f"YAHOO:{provider_name}" if provider_name else "YAHOO",
                title=content.get("title") or item.get("title") or "",
                link=link,
                published_at=(
                    content.get("pubDate") or item.get("providerPublishTime")
                ),
                description=content.get("summary"),
            ))

        return result
