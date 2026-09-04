from broker.paper import (
    PaperBroker,
)

from paper.trade_logger import (
    TradeLogger,
)


broker = PaperBroker(
    initial_cash=50_000_000
)

logger = TradeLogger()


buy_result = broker.buy(

    ticker="005930.KS",

    price=80_000,

    quantity=100,

    stop_loss=76_000,

    take_profit=86_000,

    reason="ML_PASS_BUY",
)

logger.log(
    buy_result
)


import pandas as pd


data = pd.DataFrame(
    {
        "Open": [
            80_500,
            81_500,
            83_000,
            85_000,
        ],

        "High": [
            82_000,
            84_000,
            85_500,
            87_000,
        ],

        "Low": [
            79_000,
            80_500,
            82_000,
            84_500,
        ],

        "Close": [
            81_500,
            83_000,
            84_500,
            86_500,
        ],
    },

    index=pd.date_range(
        "2026-09-01",
        periods=4,
        freq="B",
    ),
)

from paper.daily_loop import (
    DailyPaperTradingLoop,
)


loop = DailyPaperTradingLoop(
    broker=broker
)


loop.run_position_history(
    ticker="005930.KS",
    df=data,
)