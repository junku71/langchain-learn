from __future__ import annotations


def log_trade_safely(context, result, *, entity_key: str) -> bool:
    """Write the auxiliary trade CSV without breaking order state handling."""
    try:
        context.trade_logger.log(result)
    except Exception as error:
        context.store.audit(
            "TRADE_LOG_FAILED",
            entity_key,
            {"error": f"{type(error).__name__}: {error}"},
        )
        return False
    return True
