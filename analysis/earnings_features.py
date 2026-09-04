def create_earnings_features(data: dict) -> dict:
    current_eps = data.get("eps_estimate")
    eps_30d = data.get("eps_30d_ago")
    current_price = data.get("analyst_target_current")
    target_price = data.get("analyst_target_mean")
    up = data.get("eps_up_30d")
    down = data.get("eps_down_30d")
    total = up + down if up is not None and down is not None else None
    days = data.get("days_to_earnings")

    return {
        "eps_revision_30d": (
            (current_eps - eps_30d) / abs(eps_30d)
            if current_eps is not None and eps_30d not in (None, 0)
            else None
        ),
        "target_upside": (
            (target_price - current_price) / current_price
            if current_price and target_price
            else None
        ),
        "revision_balance": (
            (up - down) / total if total is not None and total > 0 else None
        ),
        "earnings_imminent": (
            0 <= days <= 7 if days is not None else None
        ),
    }
