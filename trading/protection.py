from __future__ import annotations

from datetime import datetime

from broker.models import Position
from trading.models import MinuteBar, ProtectionState


def latest_donchian_low(ticker: str, period: int = 20) -> float:
    """Return the current lower Donchian channel from daily lows."""
    from analysis.technical import get_stock_data

    frame = get_stock_data(ticker, period="6mo")
    low_column = next(
        (name for name in ("Low", "low") if name in frame.columns), None
    )
    if low_column is None or len(frame) < period:
        raise ValueError(f"Donchian 계산에 필요한 {period}일 저가가 없습니다")
    value = float(frame[low_column].tail(period).min())
    if value <= 0:
        raise ValueError("Donchian 하단값이 올바르지 않습니다")
    return value


def protection_from_position(position: Position, now: datetime) -> ProtectionState:
    highest = position.highest_price or position.avg_price
    trailing = position.trailing_stop
    if trailing is None and position.trailing_stop_pct is not None:
        trailing = highest * (1 - position.trailing_stop_pct)
    return ProtectionState(
        ticker=position.ticker,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        trailing_stop_pct=position.trailing_stop_pct,
        trailing_stop=trailing,
        highest_price=highest,
        updated_at=now,
    )


def evaluate_minute_bar(
    protection: ProtectionState,
    bar: MinuteBar,
) -> tuple[str, float | None, ProtectionState]:
    """Return action, trigger price, and updated protection state."""
    stop = protection.effective_stop()
    target = protection.take_profit
    if stop is not None and bar.open <= stop:
        return "STOP_GAP_DOWN", bar.open, protection
    if target is not None and bar.open >= target:
        return "TARGET_GAP_UP", bar.open, protection

    stop_hit = stop is not None and bar.low <= stop
    target_hit = target is not None and bar.high >= target
    if stop_hit and target_hit:
        return "STOP_AND_TARGET_SAME_BAR", stop, protection
    if stop_hit:
        reason = (
            "TRAILING_STOP"
            if protection.trailing_stop is not None and stop == protection.trailing_stop
            else "STOP_LOSS"
        )
        return reason, stop, protection
    if target_hit:
        return "TAKE_PROFIT", target, protection

    protection.highest_price = max(protection.highest_price, bar.high)
    if (
        protection.strategy == "CHANDELIER_EXIT"
        and protection.atr is not None
        and protection.atr_multiple is not None
    ):
        candidate = protection.highest_price - (
            protection.atr_multiple * protection.atr
        )
        protection.trailing_stop = max(
            protection.trailing_stop or candidate, candidate
        )
    elif protection.trailing_stop_pct is not None:
        candidate = protection.highest_price * (1 - protection.trailing_stop_pct)
        protection.trailing_stop = max(
            protection.trailing_stop or candidate, candidate
        )
    protection.updated_at = bar.timestamp
    return "HOLD", None, protection
