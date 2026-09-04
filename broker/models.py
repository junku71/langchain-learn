from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float

    sector: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None
    trailing_stop: float | None = None
    highest_price: float | None = None

    opened_at: datetime | None = None


@dataclass
class OrderResult:
    status: str

    ticker: str
    side: str

    price: float = 0.0
    quantity: int = 0

    commission: float = 0.0
    realized_pnl: float = 0.0
    order_id: str = ""

    reason: str = ""


@dataclass
class OrderExecution:
    order_id: str
    ticker: str
    side: str
    status: str
    ordered_quantity: int
    filled_quantity: int
    remaining_quantity: int
    order_price: float = 0.0
    average_fill_price: float = 0.0
    order_date: str = ""
    order_time: str = ""
    name: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Portfolio:
    initial_cash: float
    cash: float

    positions: dict[str, Position] = field(
        default_factory=dict
    )

    realized_pnl: float = 0.0
    total_commission: float = 0.0
