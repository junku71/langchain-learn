from broker.base import Broker
from paper.trade_logger import (
    TradeLogger,
)


class PositionManager:

    def __init__(
        self,
        broker: Broker,
        logger: TradeLogger,
    ):

        self.broker = broker
        self.logger = logger


    def process_daily_bar(
        self,
        ticker: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> dict:

        position = self.broker.get_position(
            ticker
        )

        if position is None:

            return {
                "action": "NO_POSITION"
            }


        fixed_stop = position.stop_loss
        trailing_stop = position.trailing_stop
        stop_candidates = [
            value
            for value in (fixed_stop, trailing_stop)
            if value is not None
        ]
        stop = max(stop_candidates) if stop_candidates else None
        target = position.take_profit
        trailing_is_active = (
            trailing_stop is not None
            and stop == trailing_stop
        )


        # -------------------------
        # Gap down
        # -------------------------

        if (
            stop is not None
            and open_price <= stop
        ):

            result = self.broker.sell(
                ticker=ticker,
                price=open_price,
                quantity=position.quantity,
                reason=(
                    "TRAILING_STOP_GAP_DOWN"
                    if trailing_is_active
                    else "STOP_GAP_DOWN"
                ),
            )

            self.logger.log(result)

            return {
                "action": "SELL",
                "reason": result.reason,
                "result": result,
            }


        # -------------------------
        # Gap up
        # -------------------------

        if (
            target is not None
            and open_price >= target
        ):

            result = self.broker.sell(
                ticker=ticker,
                price=open_price,
                quantity=position.quantity,
                reason="TARGET_GAP_UP",
            )

            self.logger.log(result)

            return {
                "action": "SELL",
                "reason": "TARGET_GAP_UP",
                "result": result,
            }


        stop_hit = (
            stop is not None
            and low_price <= stop
        )

        target_hit = (
            target is not None
            and high_price >= target
        )


        # -------------------------
        # 둘 다 터치
        # 보수적으로 Stop 우선
        # -------------------------

        if stop_hit and target_hit:

            result = self.broker.sell(
                ticker=ticker,
                price=stop,
                quantity=position.quantity,
                reason="STOP_AND_TARGET_SAME_BAR",
            )

            self.logger.log(result)

            return {
                "action": "SELL",
                "reason":
                    "STOP_AND_TARGET_SAME_BAR",
                "result": result,
            }


        # -------------------------
        # Stop Loss
        # -------------------------

        if stop_hit:

            result = self.broker.sell(
                ticker=ticker,
                price=stop,
                quantity=position.quantity,
                reason=(
                    "TRAILING_STOP"
                    if trailing_is_active
                    else "STOP_LOSS"
                ),
            )

            self.logger.log(result)

            return {
                "action": "SELL",
                "reason": result.reason,
                "result": result,
            }


        # -------------------------
        # Take Profit
        # -------------------------

        if target_hit:

            result = self.broker.sell(
                ticker=ticker,
                price=target,
                quantity=position.quantity,
                reason="TAKE_PROFIT",
            )

            self.logger.log(result)

            return {
                "action": "SELL",
                "reason": "TAKE_PROFIT",
                "result": result,
            }


        previous_highest = (
            position.highest_price
            if position.highest_price is not None
            else position.avg_price
        )
        position.highest_price = max(
            previous_highest,
            high_price,
        )

        if position.trailing_stop_pct is not None:
            next_trailing_stop = (
                position.highest_price
                * (1 - position.trailing_stop_pct)
            )
            position.trailing_stop = max(
                position.trailing_stop or next_trailing_stop,
                next_trailing_stop,
            )

        next_stop_candidates = [
            value
            for value in (
                position.stop_loss,
                position.trailing_stop,
            )
            if value is not None
        ]
        effective_stop = (
            max(next_stop_candidates)
            if next_stop_candidates
            else None
        )

        return {
            "action": "HOLD",

            "ticker": ticker,

            "close":
                close_price,

            "quantity":
                position.quantity,

            "avg_price":
                position.avg_price,

            "stop_loss":
                position.stop_loss,

            "take_profit":
                target,

            "trailing_stop":
                position.trailing_stop,

            "effective_stop":
                effective_stop,

            "highest_price":
                position.highest_price,
        }
