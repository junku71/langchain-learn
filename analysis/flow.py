from typing import Protocol

import pandas as pd

from broker.kis import KISBroker


class InvestorFlowProvider(Protocol):
    def get_investor_flow(self, ticker: str) -> list[dict]: ...


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"None of these columns found: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)


def get_investor_flow_raw(
    ticker: str,
    provider: InvestorFlowProvider | None = None,
) -> pd.DataFrame:
    data_provider = provider or KISBroker.from_env()
    return pd.DataFrame(data_provider.get_investor_flow(ticker))


def standardize_flow_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("KIS investor flow data is empty")

    date_column = find_column(
        df,
        ["stck_bsop_date", "bsop_date", "date"],
    )
    foreign_column = find_column(
        df,
        ["frgn_ntby_qty", "frgn_ntby_tr_pbmn", "frgn_net_buy_qty"],
    )
    institution_column = find_column(
        df,
        ["orgn_ntby_qty", "inst_ntby_qty", "orgn_net_buy_qty"],
    )
    result = pd.DataFrame({
        "date": pd.to_datetime(df[date_column], errors="coerce"),
        "foreign_net_buy": to_numeric_series(df[foreign_column]),
        "institution_net_buy": to_numeric_series(df[institution_column]),
    })
    return (
        result.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def calculate_flow_indicators(
    df: pd.DataFrame,
    lookback: int = 5,
) -> dict:
    if lookback <= 0:
        raise ValueError("lookback must be greater than zero")
    if df.empty:
        raise ValueError("Flow dataframe is empty")

    recent = df.tail(lookback)
    actual_days = len(recent)
    foreign_buying = recent["foreign_net_buy"] > 0
    institution_buying = recent["institution_net_buy"] > 0
    foreign_positive_days = int(foreign_buying.sum())
    institution_positive_days = int(institution_buying.sum())

    return {
        "lookback_days": actual_days,
        "foreign_net_sum": float(recent["foreign_net_buy"].sum()),
        "institution_net_sum": float(
            recent["institution_net_buy"].sum()
        ),
        "joint_buy_days": int((foreign_buying & institution_buying).sum()),
        "foreign_positive_days": foreign_positive_days,
        "institution_positive_days": institution_positive_days,
        "foreign_persistence": foreign_positive_days / actual_days,
        "institution_persistence": institution_positive_days / actual_days,
    }


def calculate_multi_window_flow(df: pd.DataFrame) -> dict:
    """Summarize a single flow snapshot across portfolio-relevant windows."""
    windows = {}
    for days in (1, 3, 5, 10, 20):
        sample = df.tail(days)
        actual = len(sample)
        foreign = sample["foreign_net_buy"]
        institution = sample["institution_net_buy"]
        windows[f"{days}d"] = {
            "available_days": actual,
            "foreign_net_sum": float(foreign.sum()),
            "institution_net_sum": float(institution.sum()),
            "combined_net_sum": float((foreign + institution).sum()),
            "foreign_positive_days": int((foreign > 0).sum()),
            "institution_positive_days": int((institution > 0).sum()),
            "joint_buy_days": int(((foreign > 0) & (institution > 0)).sum()),
            "joint_sell_days": int(((foreign < 0) & (institution < 0)).sum()),
        }
    combined_3d = windows["3d"]["combined_net_sum"]
    combined_5d = windows["5d"]["combined_net_sum"]
    combined_20d = windows["20d"]["combined_net_sum"]
    if combined_3d > 0 and combined_5d > 0 and combined_20d <= 0:
        regime = "SELLING_TO_BUYING_REVERSAL"
        momentum = "REVERSING"
    elif combined_3d < 0 and combined_5d < 0 and combined_20d >= 0:
        regime = "BUYING_TO_SELLING_REVERSAL"
        momentum = "REVERSING"
    elif combined_3d > 0 and combined_3d >= combined_5d * 0.6:
        regime, momentum = "ACCUMULATION_ACCELERATING", "ACCELERATING"
    elif combined_3d < 0 and abs(combined_3d) >= abs(combined_5d) * 0.6:
        regime, momentum = "DISTRIBUTION_ACCELERATING", "ACCELERATING"
    else:
        regime, momentum = "NO_CLEAR_CHANGE", "STABLE"
    return {
        "windows": windows,
        "flow_reversal": regime,
        "flow_momentum": momentum,
        "relative_intensity": None,
        "price_flow_relationship": "UNKNOWN",
    }


def calculate_flow_score(indicators: dict) -> dict:
    score = 50

    if indicators["foreign_net_sum"] > 0:
        score += 15
    elif indicators["foreign_net_sum"] < 0:
        score -= 15

    if indicators["institution_net_sum"] > 0:
        score += 10
    elif indicators["institution_net_sum"] < 0:
        score -= 10

    joint_days = indicators["joint_buy_days"]
    if joint_days >= 4:
        score += 15
    elif joint_days >= 2:
        score += 8

    foreign_persistence = indicators["foreign_persistence"]
    if foreign_persistence >= 0.8:
        score += 10
    elif foreign_persistence <= 0.2:
        score -= 10

    institution_persistence = indicators["institution_persistence"]
    if institution_persistence >= 0.8:
        score += 5
    elif institution_persistence <= 0.2:
        score -= 5

    score = max(0, min(100, score))
    signal = (
        "BULLISH"
        if score >= 70
        else "BEARISH"
        if score <= 35
        else "NEUTRAL"
    )
    return {**indicators, "score": score, "signal": signal}


def analyze_flow(
    ticker: str,
    lookback: int = 20,
    provider: InvestorFlowProvider | None = None,
) -> dict:
    raw_data = get_investor_flow_raw(ticker, provider)
    flow_data = standardize_flow_data(raw_data)
    indicators = calculate_flow_indicators(flow_data, lookback)
    return {
        "ticker": ticker,
        **calculate_multi_window_flow(flow_data),
        **calculate_flow_score(indicators),
    }
