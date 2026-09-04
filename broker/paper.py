from datetime import datetime

from broker.base import Broker
from broker.models import (
    Portfolio,
    Position,
    OrderResult,
)


class PaperBroker(Broker):

    def __init__(
        self,
        initial_cash: float,
        commission_rate: float = 0.00015,
    ):

        self.commission_rate = commission_rate

        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            cash=initial_cash,
        )

        self.market_prices: dict[str, float] = {}

    def set_market_price(
        self,
        ticker: str,
        price: float,
    ) -> None:

        if price <= 0:
            raise ValueError("Market price must be positive")

        self.market_prices[ticker] = price

    def get_current_price(
        self,
        ticker: str,
    ) -> float:

        if ticker in self.market_prices:
            return self.market_prices[ticker]

        position = self.get_position(ticker)

        if position is not None:
            return position.avg_price

        raise ValueError(
            f"No paper market price available for {ticker}"
        )

    # ---------------------------------
    # BUY
    # ---------------------------------

    def buy(
        self,
        ticker: str,
        price: float,
        quantity: int,
        sector: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> OrderResult:

        if price <= 0:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="BUY",
                reason="Invalid price",
            )

        if quantity <= 0:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="BUY",
                reason="Invalid quantity",
            )

        if (
            trailing_stop_pct is not None
            and not 0 < trailing_stop_pct < 1
        ):
            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="BUY",
                reason="Invalid trailing stop percentage",
            )

        self.market_prices[ticker] = price

        order_value = (
            price
            * quantity
        )

        commission = (
            order_value
            * self.commission_rate
        )

        total_cost = (
            order_value
            + commission
        )

        if total_cost > self.portfolio.cash:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="BUY",
                price=price,
                quantity=quantity,
                reason="Insufficient cash",
            )

        # 현금 감소
        self.portfolio.cash -= (
            total_cost
        )

        self.portfolio.total_commission += (
            commission
        )

        # -------------------------
        # 기존 포지션 존재
        # -------------------------

        if ticker in self.portfolio.positions:
            

            position = (
                self.portfolio
                .positions[ticker]
            )

            old_quantity = (
                position.quantity
            )

            new_quantity = (
                old_quantity
                + quantity
            )

            old_value = (
                position.avg_price
                * old_quantity
            )

            new_value = (
                price
                * quantity
            )

            position.avg_price = (
                old_value
                + new_value
            ) / new_quantity

            position.quantity = (
                new_quantity
            )

            if sector is not None:
                position.sector = sector

            if stop_loss is not None:
                position.stop_loss = (
                    stop_loss
                )

            if take_profit is not None:
                position.take_profit = (
                    take_profit
                )

            position.highest_price = max(
                position.highest_price or position.avg_price,
                price,
            )

            if trailing_stop_pct is not None:
                position.trailing_stop_pct = trailing_stop_pct
                candidate = (
                    position.highest_price
                    * (1 - trailing_stop_pct)
                )
                position.trailing_stop = max(
                    position.trailing_stop or candidate,
                    candidate,
                )

        # -------------------------
        # 신규 포지션
        # -------------------------

        else:

            self.portfolio.positions[
                ticker
            ] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=price,
                sector=sector,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
                trailing_stop=(
                    price * (1 - trailing_stop_pct)
                    if trailing_stop_pct is not None
                    else None
                ),
                highest_price=price,
                opened_at=datetime.now(),
            )

        return OrderResult(
            status="FILLED",
            ticker=ticker,
            side="BUY",
            price=price,
            quantity=quantity,
            commission=commission,
            reason=reason,
        )

    # ---------------------------------
    # SELL
    # ---------------------------------

    def sell(
        self,
        ticker: str,
        price: float,
        quantity: int,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> OrderResult:

        position = self.get_position(
            ticker
        )

        if price > 0:
            self.market_prices[ticker] = price

        if position is None:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="SELL",
                reason="Position not found",
            )

        if quantity <= 0:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="SELL",
                reason="Invalid quantity",
            )

        if quantity > position.quantity:

            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side="SELL",
                reason="Insufficient shares",
            )

        sell_value = (
            price
            * quantity
        )

        commission = (
            sell_value
            * self.commission_rate
        )

        proceeds = (
            sell_value
            - commission
        )

        gross_pnl = (
            price
            - position.avg_price
        ) * quantity

        net_pnl = (
            gross_pnl
            - commission
        )

        # 현금 증가
        self.portfolio.cash += (
            proceeds
        )

        self.portfolio.realized_pnl += (
            net_pnl
        )

        self.portfolio.total_commission += (
            commission
        )

        # 수량 감소
        position.quantity -= (
            quantity
        )

        # 전량 매도
        if position.quantity == 0:

            del self.portfolio.positions[
                ticker
            ]

        return OrderResult(
            status="FILLED",
            ticker=ticker,
            side="SELL",
            price=price,
            quantity=quantity,
            commission=commission,
            realized_pnl=net_pnl,
            reason=reason,
        )

    # ---------------------------------
    # BALANCE
    # ---------------------------------

    def get_balance(
        self
    ) -> dict:

        stock_purchase_amount = sum(
            position.avg_price * position.quantity
            for position in self.portfolio.positions.values()
        )
        stock_market_value = sum(
            self.get_current_price(ticker) * position.quantity
            for ticker, position in self.portfolio.positions.items()
        )

        return {
            "initial_cash":
                self.portfolio.initial_cash,

            "cash":
                self.portfolio.cash,

            "d2_cash":
                self.portfolio.cash,

            "stock_market_value":
                stock_market_value,

            "stock_purchase_amount":
                stock_purchase_amount,

            "total_equity":
                self.portfolio.cash + stock_market_value,

            "unrealized_pnl":
                stock_market_value - stock_purchase_amount,

            "realized_pnl":
                self.portfolio.realized_pnl,

            "total_commission":
                self.portfolio.total_commission,

            "position_count":
                len(
                    self.portfolio.positions
                ),
        }

    # ---------------------------------
    # POSITIONS
    # ---------------------------------

    def get_positions(
        self
    ) -> dict[str, Position]:

        return self.portfolio.positions.copy()

    def get_position(
        self,
        ticker: str,
    ) -> Position | None:

        return self.portfolio.positions.get(
            ticker
        )
