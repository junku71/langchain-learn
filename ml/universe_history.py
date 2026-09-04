from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd
import requests
from dotenv import load_dotenv

from ml.config import (
    DEFAULT_START,
    MARKETS,
    PREDICTION_UNIVERSE_SIZE_PER_MARKET,
    TRAIN_UNIVERSE_SIZE_PER_MARKET,
    UNIVERSE_HISTORY_PATH,
)
from ml.storage import merge_checkpoint, read_frame


class UniverseProvider(Protocol):
    def market_cap_snapshot(self, session: date) -> pd.DataFrame: ...


class KrxAPIError(RuntimeError):
    pass


@dataclass
class KrxUniverseProvider:
    api_key: str
    session: requests.Session | None = None
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "KrxUniverseProvider":
        load_dotenv()
        api_key = os.getenv("KRX_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "KRX_API_KEY is required for point-in-time market-cap universe. "
                "Register at the KRX Open API portal and add it to .env."
            )
        return cls(api_key)

    def market_cap_snapshot(self, session: date) -> pd.DataFrame:
        value = session.strftime("%Y%m%d")
        client = self.session or requests.Session()
        records = []
        for endpoint in ("stk_bydd_trd", "ksq_bydd_trd"):
            response = client.get(
                f"https://data-dbg.krx.co.kr/svc/apis/sto/{endpoint}",
                params={"basDd": value},
                headers={"AUTH_KEY": self.api_key},
                timeout=self.timeout,
            )
            if response.status_code == 401:
                raise KrxAPIError(
                    "KRX Open API rejected KRX_API_KEY (401). Confirm that this key "
                    "is active and authorized for both stk_bydd_trd and ksq_bydd_trd "
                    "daily trading services."
                )
            response.raise_for_status()
            records.extend(response.json().get("OutBlock_1", []))
        if not records:
            return pd.DataFrame()
        rows = []
        for row in records:
            market_name = str(row.get("MKT_NM", ""))
            if "KOSDAQ" in market_name:
                market = "KOSDAQ"
            elif "KOSPI" in market_name:
                market = "KOSPI"
            else:
                continue
            raw_code = str(row.get("ISU_SRT_CD") or row.get("ISU_CD") or "")
            code = raw_code.removeprefix("A")[-6:].zfill(6)
            rows.append({
                "date": pd.Timestamp(session),
                "ticker": f"{code}.{'KS' if market == 'KOSPI' else 'KQ'}",
                "name": row.get("ISU_ABBRV") or row.get("ISU_NM", ""),
                "market": market,
                "market_cap": pd.to_numeric(
                    str(row.get("MKTCAP", "")).replace(",", ""), errors="coerce"
                ),
            })
        combined = pd.DataFrame(rows).dropna(subset=["market_cap"])
        return (
            combined.sort_values(["market", "market_cap"], ascending=[True, False])
            .groupby("market", group_keys=False)
            .head(TRAIN_UNIVERSE_SIZE_PER_MARKET)
            .reset_index(drop=True)
        )


# Compatibility name for callers created during the first redesign iteration.
PykrxUniverseProvider = KrxUniverseProvider


def collect_universe_history(
    start: str | date = DEFAULT_START,
    end: str | date | None = None,
    provider: UniverseProvider | None = None,
    limit_sessions: int | None = None,
) -> pd.DataFrame:
    source = provider or KrxUniverseProvider.from_env()
    sessions = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    existing = read_frame(UNIVERSE_HISTORY_PATH)
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        counts = existing.groupby([existing["date"].dt.normalize(), "market"])[
            "ticker"
        ].nunique()
        completed = {
            session
            for session in pd.DatetimeIndex(existing["date"].dt.normalize().unique())
            if all(
                counts.get((session, market), 0) >= TRAIN_UNIVERSE_SIZE_PER_MARKET
                for market in MARKETS
            )
        }
        sessions = pd.DatetimeIndex([session for session in sessions if session not in completed])
    if limit_sessions:
        sessions = sessions[:limit_sessions]

    result = existing
    for index, session in enumerate(sessions, start=1):
        snapshot = source.market_cap_snapshot(session.date())
        if snapshot.empty:
            continue
        snapshot = snapshot.dropna(subset=["market_cap"]).sort_values(
            ["market", "market_cap"], ascending=[True, False]
        )
        snapshot["market_cap_rank"] = snapshot.groupby("market").cumcount() + 1
        snapshot["training_universe"] = snapshot["market_cap_rank"].le(
            TRAIN_UNIVERSE_SIZE_PER_MARKET
        )
        snapshot["prediction_universe"] = snapshot["market_cap_rank"].le(
            PREDICTION_UNIVERSE_SIZE_PER_MARKET
        )
        snapshot = snapshot.groupby("market", group_keys=False).head(
            TRAIN_UNIVERSE_SIZE_PER_MARKET
        )
        result = merge_checkpoint(snapshot, UNIVERSE_HISTORY_PATH, ["date", "ticker"])
        print(f"[{index}/{len(sessions)}] {session.date()}: {len(snapshot)} universe rows")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect point-in-time market-cap universe")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end")
    parser.add_argument("--limit-sessions", type=int)
    args = parser.parse_args()
    result = collect_universe_history(args.start, args.end, limit_sessions=args.limit_sessions)
    print(f"Saved {len(result):,} rows to {UNIVERSE_HISTORY_PATH}")


if __name__ == "__main__":
    main()
