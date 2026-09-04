import math
from typing import Protocol

from broker.kis import KISBroker


class FundamentalDataProvider(Protocol):
    def get_fundamental_data(self, ticker: str) -> dict: ...


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None

    number = float(value)
    return number if math.isfinite(number) else None


def calculate_fundamental_score(
    per: float | None,
    pbr: float | None,
    roe: float | None,
    debt_ratio: float | None,
    pcr: float | None = None,
    ev_ebitda: float | None = None,
    revenue_growth: float | None = None,
    operating_profit_growth: float | None = None,
) -> dict:
    values = {
        "PER": _finite_or_none(per),
        "PBR": _finite_or_none(pbr),
        "PCR": _finite_or_none(pcr),
        "EV_EBITDA": _finite_or_none(ev_ebitda),
        "ROE": _finite_or_none(roe),
        "debt_ratio": _finite_or_none(debt_ratio),
        "revenue_growth": _finite_or_none(revenue_growth),
        "operating_profit_growth": _finite_or_none(operating_profit_growth),
    }
    weights = {
        "PER": 15,
        "PBR": 10,
        "PCR": 10,
        "EV_EBITDA": 10,
        "ROE": 15,
        "debt_ratio": 10,
        "revenue_growth": 15,
        "operating_profit_growth": 15,
    }

    def lower_is_better(value, levels):
        if value <= 0:
            return 0.0
        for ceiling, points in levels:
            if value <= ceiling:
                return points
        return 0.0

    def higher_is_better(value, levels):
        for floor, points in levels:
            if value >= floor:
                return points
        return 0.0

    scorers = {
        "PER": lambda value: lower_is_better(value, [(10, 100), (20, 70), (30, 40)]),
        "PBR": lambda value: lower_is_better(value, [(1, 100), (2, 75), (4, 40)]),
        "PCR": lambda value: lower_is_better(value, [(8, 100), (15, 70), (25, 40)]),
        "EV_EBITDA": lambda value: lower_is_better(value, [(6, 100), (10, 70), (15, 40)]),
        "ROE": lambda value: higher_is_better(value, [(15, 100), (10, 75), (5, 50), (0, 25)]),
        "debt_ratio": lambda value: lower_is_better(value, [(50, 100), (100, 75), (150, 50), (200, 25)]),
        "revenue_growth": lambda value: higher_is_better(value, [(20, 100), (10, 75), (0, 50), (-10, 25)]),
        "operating_profit_growth": lambda value: higher_is_better(value, [(25, 100), (10, 75), (0, 50), (-20, 25)]),
    }
    component_scores = {
        name: (scorers[name](value) if value is not None else None)
        for name, value in values.items()
    }
    available_weight = sum(
        weights[name] for name, value in component_scores.items() if value is not None
    )
    raw_score = (
        round(
            sum(
                component_scores[name] * weights[name]
                for name in component_scores
                if component_scores[name] is not None
            ) / available_weight,
            1,
        )
        if available_weight
        else 50.0
    )

    coverage_pct = round(available_weight, 1)
    # Pull sparse results toward neutral so one attractive multiple cannot
    # masquerade as a high-conviction fundamental thesis.
    score = round(50 + (raw_score - 50) * coverage_pct / 100, 1)
    confidence = (
        "HIGH" if coverage_pct >= 80
        else "MEDIUM" if coverage_pct >= 50
        else "LOW"
    )

    def average(names: list[str]) -> float | None:
        available = [component_scores[name] for name in names if component_scores[name] is not None]
        return round(sum(available) / len(available), 1) if available else None

    category_scores = {
        "valuation": average(["PER", "PBR", "PCR", "EV_EBITDA"]),
        "profitability": average(["ROE"]),
        "earnings": average(["revenue_growth", "operating_profit_growth"]),
        "financial_quality": None,
        "balance_sheet": average(["debt_ratio"]),
        "relative_valuation": None,
    }

    if score >= 70:
        signal = "BULLISH"
    elif score <= 40:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "score": score,
        "raw_score": raw_score,
        "signal": signal,
        "confidence": confidence,
        **values,
        "component_scores": component_scores,
        "category_scores": category_scores,
        "data_coverage_pct": coverage_pct,
    }


def get_fundamental_data(
    ticker: str,
    provider: FundamentalDataProvider | None = None,
) -> dict:
    data_provider = provider or KISBroker.from_env()
    return data_provider.get_fundamental_data(ticker)


def analyze_fundamental(
    ticker: str,
    provider: FundamentalDataProvider | None = None,
) -> dict:
    data = get_fundamental_data(ticker, provider)
    result = calculate_fundamental_score(
        per=data.get("PER"),
        pbr=data.get("PBR"),
        roe=data.get("ROE"),
        debt_ratio=data.get("debt_ratio"),
        pcr=data.get("PCR"),
        ev_ebitda=data.get("EV_EBITDA"),
        revenue_growth=data.get("revenue_growth"),
        operating_profit_growth=data.get("operating_profit_growth"),
    )
    result["ticker"] = ticker
    return result
