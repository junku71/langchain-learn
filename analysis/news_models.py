from dataclasses import asdict, dataclass


@dataclass
class NewsItem:
    source: str
    title: str
    link: str | None = None
    published_at: str | int | None = None
    description: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EarningsEvent:
    earnings_date: str | None = None
    days_to_earnings: int | None = None
    eps_estimate: float | None = None
    eps_low: float | None = None
    eps_high: float | None = None
    eps_analysts: int | None = None
    revenue_estimate: float | None = None
    revenue_low: float | None = None
    revenue_high: float | None = None
    revenue_analysts: int | None = None
    eps_7d_ago: float | None = None
    eps_30d_ago: float | None = None
    eps_up_30d: int | None = None
    eps_down_30d: int | None = None
    analyst_target_current: float | None = None
    analyst_target_mean: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
