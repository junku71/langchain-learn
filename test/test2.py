from broker.paper import PaperBroker
from portfolio_manager import PortfolioManager


paper_broker = PaperBroker(
    initial_cash=50_000_000
)


paper_broker.buy(
    ticker="005930.KS",
    price=80_000,
    quantity=100,
    sector="SEMICONDUCTOR",
    stop_loss=76_000,
    take_profit=86_000,
)

paper_broker.buy(
    ticker="000660.KS",
    price=250_000,
    quantity=30,
    sector="SEMICONDUCTOR",
    stop_loss=235_000,
    take_profit=275_000,
)

paper_broker.buy(
    ticker="012450.KS",
    price=500_000,
    quantity=10,
    sector="DEFENSE",
    stop_loss=470_000,
    take_profit=550_000,
)

current_prices = {

    "005930.KS":
        84_000,

    "000660.KS":
        260_000,

    "012450.KS":
        490_000,
}

portfolio_manager = (
    PortfolioManager(
        paper_broker
    )
)


report = (
    portfolio_manager
    .evaluate(
        current_prices
    )
)


print(report)
