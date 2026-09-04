from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ml.config import (
    FEATURE_PANEL_PATH,
    QUALITY_REPORT_PATH,
    TRAIN_UNIVERSE_SIZE,
    TRAIN_UNIVERSE_SIZE_PER_MARKET,
)
from ml.features import (
    CROSS_SECTION_FEATURES,
    FEATURE_COLUMNS,
    FLOW_FEATURES,
    FUNDAMENTAL_FEATURES,
    MARKET_FEATURES,
    MOMENTUM_FEATURES,
    TECHNICAL_FEATURES,
    VOLUME_FEATURES,
)
from ml.storage import read_frame


GROUPS = {
    "momentum": MOMENTUM_FEATURES,
    "technical": TECHNICAL_FEATURES,
    "volume": VOLUME_FEATURES,
    "flow": FLOW_FEATURES,
    "fundamental": FUNDAMENTAL_FEATURES,
    "market": MARKET_FEATURES,
    "cross_section": CROSS_SECTION_FEATURES,
}
MINIMUM_GROUP_COVERAGE = {
    "momentum": 0.70,
    "technical": 0.70,
    "volume": 0.70,
    "flow": 0.40,
    "fundamental": 0.40,
    "market": 0.50,
    "cross_section": 0.50,
}


def validate_dataset(
    frame: pd.DataFrame | None = None,
    *,
    fail: bool = True,
    require_training_ready: bool = False,
    write_report: bool = True,
) -> dict:
    panel = read_frame(FEATURE_PANEL_PATH) if frame is None else frame.copy()
    required = {"date", "ticker", "market_cap_rank", *FEATURE_COLUMNS}
    missing_columns = sorted(required - set(panel))
    duplicate_rows = int(panel.duplicated(["date", "ticker"]).sum()) if not panel.empty else 0
    infinite_values = int(np.isinf(panel[[c for c in FEATURE_COLUMNS if c in panel]].to_numpy(dtype=float)).sum()) if not panel.empty else 0
    daily_size = panel.groupby("date")["ticker"].nunique() if not panel.empty else pd.Series(dtype=float)
    daily_market_size = (
        panel.groupby(["date", "market"])["ticker"].nunique()
        if not panel.empty and "market" in panel
        else pd.Series(dtype=float)
    )
    coverage = {
        column: round(float(panel[column].notna().mean()), 4)
        for column in FEATURE_COLUMNS if column in panel
    }
    group_coverage = {
        name: round(float(np.mean([coverage.get(column, 0.0) for column in columns])), 4)
        for name, columns in GROUPS.items()
    }
    report = {
        "rows": int(len(panel)),
        "tickers": int(panel["ticker"].nunique()) if "ticker" in panel else 0,
        "first_date": str(pd.to_datetime(panel["date"]).min().date()) if not panel.empty else None,
        "last_date": str(pd.to_datetime(panel["date"]).max().date()) if not panel.empty else None,
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "infinite_values": infinite_values,
        "max_daily_universe": int(daily_size.max()) if not daily_size.empty else 0,
        "max_daily_market_universe": (
            int(daily_market_size.max()) if not daily_market_size.empty else 0
        ),
        "feature_coverage": coverage,
        "group_coverage": group_coverage,
    }
    errors = []
    if missing_columns: errors.append("missing columns")
    if duplicate_rows: errors.append("duplicate date/ticker rows")
    if infinite_values: errors.append("infinite feature values")
    if report["max_daily_universe"] > TRAIN_UNIVERSE_SIZE: errors.append("training universe exceeds limit")
    if report["max_daily_market_universe"] > TRAIN_UNIVERSE_SIZE_PER_MARKET:
        errors.append("per-market training universe exceeds limit")
    low_coverage = sorted(column for column, ratio in coverage.items() if ratio < 0.5)
    training_blockers = [
        f"{name} coverage {group_coverage[name]:.1%} < {minimum:.1%}"
        for name, minimum in MINIMUM_GROUP_COVERAGE.items()
        if group_coverage[name] < minimum
    ]
    report["training_ready"] = not training_blockers
    report["training_blockers"] = training_blockers
    report["warnings"] = [
        f"feature coverage below 50%: {', '.join(low_coverage)}"
    ] if low_coverage else []
    if require_training_ready and training_blockers:
        errors.append("insufficient source-group coverage")
        report["valid"] = False
        report["errors"] = errors
    report["valid"] = not errors
    report["errors"] = errors
    if write_report:
        QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if fail and errors:
        raise ValueError(f"Dataset validation failed: {', '.join(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ML dataset")
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(
        require_training_ready=not args.structural_only
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
