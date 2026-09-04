import pandas as pd
import yfinance as yf

from analysis.news_models import EarningsEvent


def safe_float(value) -> float | None:
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int | None:
    try:
        return None if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def get_next_earnings_date(stock: yf.Ticker):
    try:
        calendar = stock.calendar
        dates = calendar.get("Earnings Date") or calendar.get("earningsDate")
    except Exception:
        return None

    if dates is None:
        return None
    dates = dates if isinstance(dates, (list, tuple)) else [dates]
    parsed = [pd.Timestamp(value) for value in dates]
    parsed = [
        value.tz_convert("Asia/Seoul").tz_localize(None)
        if value.tzinfo is not None
        else value
        for value in parsed
    ]
    today = pd.Timestamp.now().normalize()
    future = [value for value in parsed if value.normalize() >= today]
    return min(future) if future else None


def _quarter_row(stock: yf.Ticker, method: str):
    try:
        data = getattr(stock, method)()
        return data.loc["0q"] if data is not None and "0q" in data.index else None
    except Exception:
        return None


def get_earnings_consensus(ticker: str) -> EarningsEvent:
    stock = yf.Ticker(ticker)
    result = EarningsEvent()
    earnings_date = get_next_earnings_date(stock)

    if earnings_date is not None:
        result.earnings_date = earnings_date.strftime("%Y-%m-%d")
        result.days_to_earnings = int(
            (earnings_date.normalize() - pd.Timestamp.now().normalize()).days
        )

    eps = _quarter_row(stock, "get_earnings_estimate")
    if eps is not None:
        result.eps_estimate = safe_float(eps.get("avg"))
        result.eps_low = safe_float(eps.get("low"))
        result.eps_high = safe_float(eps.get("high"))
        result.eps_analysts = safe_int(eps.get("numberOfAnalysts"))

    revenue = _quarter_row(stock, "get_revenue_estimate")
    if revenue is not None:
        result.revenue_estimate = safe_float(revenue.get("avg"))
        result.revenue_low = safe_float(revenue.get("low"))
        result.revenue_high = safe_float(revenue.get("high"))
        result.revenue_analysts = safe_int(revenue.get("numberOfAnalysts"))

    trend = _quarter_row(stock, "get_eps_trend")
    if trend is not None:
        result.eps_7d_ago = safe_float(trend.get("7daysAgo"))
        result.eps_30d_ago = safe_float(trend.get("30daysAgo"))

    revision = _quarter_row(stock, "get_eps_revisions")
    if revision is not None:
        result.eps_up_30d = safe_int(revision.get("upLast30days"))
        result.eps_down_30d = safe_int(revision.get("downLast30days"))

    try:
        targets = stock.get_analyst_price_targets() or {}
        result.analyst_target_current = safe_float(targets.get("current"))
        result.analyst_target_mean = safe_float(targets.get("mean"))
    except Exception:
        pass

    return result


class YahooEarningsProvider:
    def get_consensus(self, ticker: str) -> EarningsEvent:
        return get_earnings_consensus(ticker)
