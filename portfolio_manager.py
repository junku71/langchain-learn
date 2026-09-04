from dataclasses import dataclass

from broker.base import Broker


@dataclass(frozen=True)
class PortfolioLimits:
    max_position_pct: float = 0.20
    max_sector_pct: float = 0.30
    max_invested_pct: float = 0.80


def can_add_position(
    portfolio_report: dict,
    sector: str,
    new_position_value: float,
    ticker: str | None = None,
    limits: PortfolioLimits | None = None,
) -> dict:

    limits = limits or PortfolioLimits()
    total_equity = portfolio_report["total_equity"]

    if total_equity <= 0:
        return {
            "approved": False,
            "reason": "Invalid equity",
        }

    if new_position_value <= 0:
        return {
            "approved": False,
            "reason": "Invalid position value",
        }

    existing_position_value = 0.0

    if ticker is not None:
        existing_position = portfolio_report[
            "positions"
        ].get(ticker)

        if existing_position is not None:
            existing_position_value = existing_position[
                "market_value"
            ]

    new_position_pct = new_position_value / total_equity
    projected_position_pct = (
        existing_position_value + new_position_value
    ) / total_equity

    existing_sector_pct = portfolio_report[
        "sector_exposure"
    ].get(sector, 0.0)
    projected_sector_pct = (
        existing_sector_pct + new_position_pct
    )

    projected_invested_pct = (
        portfolio_report["market_value"] + new_position_value
    ) / total_equity

    result = {
        "approved": True,
        "new_position_pct": new_position_pct,
        "projected_position_pct": projected_position_pct,
        "projected_sector_pct": projected_sector_pct,
        "projected_invested_pct": projected_invested_pct,
    }

    if projected_position_pct > limits.max_position_pct:
        return {
            **result,
            "approved": False,
            "reason": "Position limit exceeded",
        }

    if projected_sector_pct > limits.max_sector_pct:
        return {
            **result,
            "approved": False,
            "reason": "Sector limit exceeded",
        }

    if projected_invested_pct > limits.max_invested_pct:
        return {
            **result,
            "approved": False,
            "reason": "Invested capital limit exceeded",
        }

    return result


class PortfolioManager:

    def __init__(
        self,
        broker: Broker,
    ):

        self.broker = broker


    def evaluate(
        self,
        current_prices: dict[str, float],
    ) -> dict:

        positions = (
            self.broker
            .get_positions()
        )

        balance = (
            self.broker
            .get_balance()
        )

        cash = balance[
            "cash"
        ]

        total_market_value = 0.0
        total_unrealized_pnl = 0.0

        position_details = {}

        sector_values = {}


        for (
            ticker,
            position
        ) in positions.items():

            current_price = (
                current_prices.get(
                    ticker
                )
            )

            if current_price is None:
                continue

            market_value = (
                current_price
                * position.quantity
            )

            cost_value = (
                position.avg_price
                * position.quantity
            )

            unrealized_pnl = (
                market_value
                - cost_value
            )

            unrealized_pct = (
                unrealized_pnl
                / cost_value
                if cost_value > 0
                else 0.0
            )

            total_market_value += (
                market_value
            )

            total_unrealized_pnl += (
                unrealized_pnl
            )

            sector = (
                position.sector
                or "UNKNOWN"
            )

            sector_values[
                sector
            ] = (
                sector_values.get(
                    sector,
                    0.0
                )
                + market_value
            )

            position_details[
                ticker
            ] = {

                "quantity":
                    position.quantity,

                "avg_price":
                    position.avg_price,

                "current_price":
                    current_price,

                "market_value":
                    market_value,

                "unrealized_pnl":
                    unrealized_pnl,

                "unrealized_pct":
                    unrealized_pct,

                "sector":
                    sector,

                "stop_loss":
                    position.stop_loss,

                "take_profit":
                    position.take_profit,

                "trailing_stop_pct":
                    position.trailing_stop_pct,

                "trailing_stop":
                    position.trailing_stop,

                "highest_price":
                    position.highest_price,
            }


        total_equity = (
            cash
            + total_market_value
        )


        # 종목별 비중

        for (
            ticker,
            detail
        ) in position_details.items():

            if total_equity > 0:

                detail[
                    "portfolio_weight"
                ] = (
                    detail[
                        "market_value"
                    ]
                    / total_equity
                )

            else:

                detail[
                    "portfolio_weight"
                ] = 0.0


        # 섹터별 비중

        sector_exposure = {}

        for (
            sector,
            value
        ) in sector_values.items():

            if total_equity > 0:

                sector_exposure[
                    sector
                ] = (
                    value
                    / total_equity
                )

            else:

                sector_exposure[
                    sector
                ] = 0.0


        return {

            "cash":
                cash,

            "market_value":
                total_market_value,

            "total_equity":
                total_equity,

            "realized_pnl":
                balance[
                    "realized_pnl"
                ],

            "unrealized_pnl":
                total_unrealized_pnl,

            "total_pnl":
                (
                    balance[
                        "realized_pnl"
                    ]
                    + total_unrealized_pnl
                ),

            "positions":
                position_details,

            "sector_exposure":
                sector_exposure,
        }
