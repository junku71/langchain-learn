import math

from risk.risk_config import (
    RiskConfig,
)

def calculate_position_risk(
    price: float,
    atr: float,
    account_size: float,
    config: RiskConfig,
) -> dict:

    stop_distance = (
        atr
        * config.atr_stop_multiple
    )

    stop_loss = (
        price
        - stop_distance
    )

    risk_per_share = (
        price
        - stop_loss
    )

    risk_amount = (
        account_size
        * config.risk_per_trade
    )

    if risk_per_share <= 0:

        return {
            "approved": False,
            "reason":
                "Invalid risk per share",
        }

    risk_based_shares = (
        risk_amount
        / risk_per_share
    )

    max_position_value = (
        account_size
        * config.max_position_pct
    )

    position_based_shares = (
        max_position_value
        / price
    )

    position_size = math.floor(
        min(
            risk_based_shares,
            position_based_shares,
        )
    )

    position_value = (
        position_size
        * price
    )

    actual_risk = (
        position_size
        * risk_per_share
    )

    take_profit = (
        price
        + (
            risk_per_share
            * config.reward_risk_ratio
        )
    )

    return {
        "approved":
            position_size > 0,

        "price":
            price,

        "atr":
            atr,

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "risk_per_share":
            risk_per_share,

        "risk_amount_limit":
            risk_amount,

        "position_size":
            position_size,

        "position_value":
            position_value,

        "actual_risk":
            actual_risk,

        "position_pct":
            position_value
            / account_size,
    }