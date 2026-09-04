import pandas as pd

from broker.base import Broker
from portfolio_manager import PortfolioManager

from paper.position_manager import (
    PositionManager,
)

from paper.trade_logger import (
    TradeLogger,
)


class DailyPaperTradingLoop:

    def __init__(
        self,
        broker: Broker,
        logger: TradeLogger | None = None,
    ):

        self.broker = broker

        self.logger = logger or TradeLogger()

        self.portfolio_manager = PortfolioManager(
            broker
        )

        self.last_prices: dict[str, float] = {}

        self.position_manager = (
            PositionManager(
                broker=broker,
                logger=self.logger,
            )
        )

    def run_day(
        self,
        date,
        bars: dict,
    ) -> dict:

        actions = {}

        # A position can be removed during iteration, so use a snapshot.
        positions = self.broker.get_positions()

        for ticker in positions:

            bar = bars.get(ticker)

            if bar is None:
                actions[ticker] = {
                    "action": "NO_DATA",
                }
                continue

            close_price = float(bar["Close"])
            self.last_prices[ticker] = close_price

            actions[ticker] = (
                self.position_manager.process_daily_bar(
                    ticker=ticker,
                    open_price=float(bar["Open"]),
                    high_price=float(bar["High"]),
                    low_price=float(bar["Low"]),
                    close_price=close_price,
                )
            )

        current_prices = {}

        for ticker, position in (
            self.broker.get_positions().items()
        ):
            current_prices[ticker] = self.last_prices.get(
                ticker,
                position.avg_price,
            )

        portfolio = self.portfolio_manager.evaluate(
            current_prices
        )

        return {
            "date": date,
            "actions": actions,
            "portfolio": portfolio,
        }

    def run_portfolio_history(
        self,
        price_history: dict[str, pd.DataFrame],
    ) -> list[dict]:

        dates = sorted({
            date
            for df in price_history.values()
            for date in df.index
        })
        daily_results = []

        for date in dates:

            if not self.broker.get_positions():
                break

            bars = {
                ticker: df.loc[date]
                for ticker, df in price_history.items()
                if date in df.index
            }

            daily_results.append(
                self.run_day(
                    date=date,
                    bars=bars,
                )
            )

        return daily_results

    def run_position_history(
        self,
        ticker: str,
        df: pd.DataFrame,
    ):

        for date, row in df.iterrows():

            # 이미 매도되어
            # 포지션이 없어졌다면 종료
            if (
                self.broker
                .get_position(ticker)
                is None
            ):

                print(
                    date,
                    "Position closed"
                )

                break

            result = (
                self.position_manager
                .process_daily_bar(

                    ticker=ticker,

                    open_price=float(
                        row["Open"]
                    ),

                    high_price=float(
                        row["High"]
                    ),

                    low_price=float(
                        row["Low"]
                    ),

                    close_price=float(
                        row["Close"]
                    ),
                )
            )

            print(
                date,
                result["action"],
                result.get(
                    "reason",
                    ""
                ),
            )
