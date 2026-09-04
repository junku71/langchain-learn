from __future__ import annotations

import argparse

import pandas as pd

from ml.build_dataset import build_dataset
from ml.collect_data import collect_data
from ml.config import DEFAULT_START, FEATURE_PANEL_PATH
from ml.validate_dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect, build and validate the point-in-time ML dataset"
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end")
    parser.add_argument("--years", type=int, help="Compatibility alias for --start")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--limit", type=int, help="Compatibility alias for --limit-sessions")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-fundamental", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    start = args.start
    if start is None and args.years:
        start = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)).date())
    start = start or DEFAULT_START
    limit_sessions = args.limit_sessions or args.limit

    if not args.skip_collect:
        collect_data(
            start,
            args.end,
            limit_sessions=limit_sessions,
            skip_flow=args.skip_flow,
            skip_fundamental=args.skip_fundamental,
        )
    features = build_dataset()
    report = validate_dataset(
        features,
        require_training_ready=limit_sessions is None,
    )
    print(
        f"Prepared {len(features):,} rows, {features['ticker'].nunique():,} tickers "
        f"at {FEATURE_PANEL_PATH}"
    )
    print(
        f"Dataset valid={report['valid']}; training_ready={report['training_ready']}; "
        f"warnings={len(report['warnings'])}"
    )


if __name__ == "__main__":
    main()
