from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from broker.kis import KISAPIError, KISBroker
from ml.panel_data import FLOW_HISTORY_PATH, load_universe


COLUMNS = ["date", "ticker", "foreign_net", "institution_net"]


def _read_existing(path: Path) -> pd.DataFrame:
    frames = []
    for candidate in (path, path.with_suffix(".pending.csv")):
        if candidate.exists():
            frames.append(pd.read_csv(candidate, dtype={"ticker": str}, parse_dates=["date"]))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    return frame[COLUMNS].drop_duplicates(["date", "ticker"], keep="last")


def _save_checkpoint(frame: pd.DataFrame, path: Path) -> bool:
    temporary = path.with_suffix(".tmp.csv")
    pending = path.with_suffix(".pending.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    try:
        temporary.replace(path)
        pending.unlink(missing_ok=True)
        return True
    except PermissionError:
        temporary.replace(pending)
        print(f"Flow CSV is locked; checkpoint preserved at {pending}")
        return False


def collect_flow_history(
    years: int = 5,
    output_path: Path = FLOW_HISTORY_PATH,
    limit: int | None = None,
    broker: KISBroker | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    universe = (
        pd.DataFrame({"ticker": tickers})
        if tickers is not None
        else load_universe(update_sectors=False)
    )
    if limit:
        universe = universe.head(limit)
    provider = broker or KISBroker.from_env()
    existing = _read_existing(output_path)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for index, ticker in enumerate(universe["ticker"], start=1):
        known = existing.loc[existing["ticker"].eq(ticker), "date"]
        ranges = []
        if known.empty:
            ranges.append((start, end))
        else:
            if known.min() > start:
                ranges.append((start, known.min() - pd.Timedelta(days=1)))
            if known.max() < end:
                ranges.append((known.max() + pd.Timedelta(days=1), end))

        new_rows = []
        for range_start, range_end in ranges:
            if range_start <= range_end:
                try:
                    new_rows.extend(provider.get_investor_flow_history(
                        ticker,
                        range_start.to_pydatetime(),
                        range_end.to_pydatetime(),
                    ))
                except KISAPIError as error:
                    if "OPSQ2001" not in str(error):
                        raise
                    # KIS makes this quotation endpoint unavailable after its
                    # daily cutoff. Preserve completed checkpoints and let the
                    # rest of the data preparation continue.
                    if not output_path.exists():
                        _save_checkpoint(existing, output_path)
                    print(
                        "KIS 수급 이력 API 운영시간(00:00~15:40)이 아닙니다. "
                        "수급 수집을 보류하고 다음 단계로 진행합니다."
                    )
                    print(
                        "허용 시간에 같은 명령을 다시 실행하면 기존 CSV 이후부터 "
                        "수집을 재개합니다."
                    )
                    return existing.reset_index(drop=True)
        if new_rows:
            addition = pd.DataFrame(new_rows)
            addition["date"] = pd.to_datetime(addition["date"])
            existing = pd.concat([existing, addition], ignore_index=True)
            existing = existing.drop_duplicates(["date", "ticker"], keep="last")
            existing = existing.sort_values(["date", "ticker"])
            if not _save_checkpoint(existing, output_path):
                return existing.reset_index(drop=True)
        print(f"[{index}/{len(universe)}] {ticker}: {len(new_rows)} rows added")

    return existing.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect five years of KIS investor flows")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = collect_flow_history(years=args.years, limit=args.limit)
    print(f"Saved {len(result):,} rows to {FLOW_HISTORY_PATH}")


if __name__ == "__main__":
    main()
