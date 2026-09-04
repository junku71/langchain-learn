import os

from dotenv import load_dotenv

from broker.base import Broker
from broker.kis import KISBroker
from broker.paper import PaperBroker
from paper.trade_logger import TradeLogger


load_dotenv()


def create_broker(
    broker_type: str | None = None,
) -> Broker:

    selected = (
        broker_type
        or os.getenv("BROKER_TYPE", "paper")
    ).strip().lower()
    market_region = os.getenv("TRADING_MARKET_REGION", "KR").strip().upper()

    if selected == "paper":
        return PaperBroker(
            initial_cash=float(
                os.getenv("PAPER_INITIAL_CASH", "50000000")
            ),
            commission_rate=float(
                os.getenv("PAPER_COMMISSION_RATE", "0.00015")
            ),
        )

    if selected == "kis":
        if market_region == "US":
            raise ValueError(
                "KISBroker currently uses domestic-stock endpoints. "
                "Use BROKER_TYPE=paper in US mode until an overseas broker adapter is configured."
            )
        return KISBroker.from_env()

    raise ValueError(
        "BROKER_TYPE must be either 'paper' or 'kis'"
    )


broker: Broker = create_broker()

trade_logger = TradeLogger(
    filename="logs/trades.csv"
)
