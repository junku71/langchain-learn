from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class MinuteBar:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    source: str = "UNKNOWN"

    def valid(self) -> bool:
        return (
            min(self.open, self.high, self.low, self.close) > 0
            and self.high >= self.low
        )


@dataclass(frozen=True)
class Candidate:
    ticker: str
    market: str
    sector: str
    ml_score: float
    classification_probability: float
    ml_rank: int
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtectionState:
    ticker: str
    stop_loss: float | None
    take_profit: float | None
    trailing_stop_pct: float | None
    trailing_stop: float | None
    highest_price: float
    updated_at: datetime
    strategy: str = "LEGACY"
    atr: float | None = None
    atr_multiple: float | None = None
    donchian_period: int | None = None

    def effective_stop(self) -> float | None:
        values = [
            value for value in (self.stop_loss, self.trailing_stop)
            if value is not None
        ]
        return max(values) if values else None
