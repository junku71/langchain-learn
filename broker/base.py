from abc import ABC, abstractmethod

from broker.models import (
    OrderExecution,
    OrderResult,
    Position,
)


class Broker(ABC):

    @abstractmethod
    def get_current_price(
        self,
        ticker: str,
    ) -> float:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def sell(
        self,
        ticker: str,
        price: float,
        quantity: int,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> OrderResult:
        pass

    @abstractmethod
    def get_balance(self) -> dict:
        pass

    @abstractmethod
    def get_positions(
        self
    ) -> dict[str, Position]:
        pass

    @abstractmethod
    def get_position(
        self,
        ticker: str,
    ) -> Position | None:
        pass

    def get_order_execution(
        self,
        order_id: str,
        order_date: str,
        ticker: str | None = None,
    ) -> OrderExecution | None:
        """Return the latest broker execution state when supported."""
        raise NotImplementedError("Order reconciliation is not supported")

    def list_order_executions(self, order_date: str) -> list[OrderExecution]:
        """Return all broker orders/executions for a trading date when supported."""
        raise NotImplementedError("Daily order history is not supported")

    def cancel_order(
        self,
        order_id: str,
        order_date: str,
        ticker: str | None = None,
    ) -> OrderResult:
        """Cancel all remaining quantity of an open order when supported."""
        raise NotImplementedError("Order cancellation is not supported")
