from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from broker.kis import KISBroker


SECTOR_CACHE_PATH = Path("data/ml/kis_sector_cache.csv")
UNKNOWN_VALUES = {"", "UNKNOWN", "NONE", "NAN"}


class SectorProvider(Protocol):
    def get_stock_sector(self, ticker: str) -> dict: ...


def _is_unknown(value) -> bool:
    return pd.isna(value) or str(value).strip().upper() in UNKNOWN_VALUES


def _stock_code(value: str) -> str:
    return str(value).strip().split(".", 1)[0].replace(".0", "").zfill(6)


def _load_cache(path: Path) -> pd.DataFrame:
    columns = ["code", "kis_sector", "kis_market", "updated_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    cache = pd.read_csv(path, dtype={"code": str}, keep_default_na=False)
    for column in columns:
        if column not in cache:
            cache[column] = ""
    cache["code"] = cache["code"].map(_stock_code)
    return cache[columns].drop_duplicates("code", keep="last")


def _save_cache(cache: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    cache.sort_values("code").to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def update_universe_sectors(
    universe: pd.DataFrame,
    provider: SectorProvider | None = None,
    cache_path: str | Path = SECTOR_CACHE_PATH,
    *,
    checkpoint_every: int = 10,
    retry_unknown_after_days: int = 30,
) -> pd.DataFrame:
    if "ticker" not in universe:
        raise ValueError("Universe needs a ticker column")
    result = universe.copy()
    result["code"] = result["ticker"].map(_stock_code)
    if "kis_sector" not in result:
        result["kis_sector"] = "UNKNOWN"
    if "kis_market" not in result:
        result["kis_market"] = ""
    if "ml_sector" not in result:
        result["ml_sector"] = result.get("sector", "UNKNOWN")

    path = Path(cache_path)
    cache = _load_cache(path)
    cache_map = cache.set_index("code").to_dict("index")
    for index, row in result.iterrows():
        cached = cache_map.get(row["code"], {})
        if _is_unknown(row["kis_sector"]) and not _is_unknown(
            cached.get("kis_sector")
        ):
            result.at[index, "kis_sector"] = cached["kis_sector"]
            result.at[index, "kis_market"] = cached.get("kis_market", "")

    now = pd.Timestamp.now(tz="UTC")
    recently_attempted = set()
    for _, row in cache.iterrows():
        if not _is_unknown(row["kis_sector"]):
            continue
        attempted = pd.to_datetime(row["updated_at"], utc=True, errors="coerce")
        if pd.notna(attempted) and now - attempted < pd.Timedelta(
            days=retry_unknown_after_days
        ):
            recently_attempted.add(row["code"])
    missing = result.loc[
        result["kis_sector"].map(_is_unknown)
        & ~result["code"].isin(recently_attempted),
        "ticker",
    ].tolist()
    data_provider = provider or (KISBroker.from_env() if missing else None)
    new_rows = []
    for number, ticker in enumerate(missing, 1):
        try:
            item = data_provider.get_stock_sector(ticker)
        except Exception as error:
            print(f"[{number:3d}/{len(missing)}] {ticker}: {error}")
            continue
        item["updated_at"] = datetime.now(timezone.utc).isoformat()
        new_rows.append(item)
        code = _stock_code(ticker)
        if not _is_unknown(item.get("kis_sector")):
            mask = result["code"].eq(code)
            result.loc[mask, "kis_sector"] = item["kis_sector"]
            result.loc[mask, "kis_market"] = item.get("kis_market") or ""
        print(f"[{number:3d}/{len(missing)}] {ticker}: {item['kis_sector']}")
        if checkpoint_every > 0 and len(new_rows) % checkpoint_every == 0:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache = cache.drop_duplicates("code", keep="last")
            _save_cache(cache, path)
            new_rows = []

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates("code", keep="last")
        _save_cache(cache, path)

    fill_ml = result["ml_sector"].map(_is_unknown)
    result.loc[fill_ml, "ml_sector"] = result.loc[fill_ml, "kis_sector"]
    foreign_listed = (
        result["ml_sector"].map(_is_unknown)
        & result["code"].str.startswith("9")
    )
    result.loc[foreign_listed, "ml_sector"] = "FOREIGN_LISTED"
    result["sector"] = result["ml_sector"]
    return result.drop(columns="code")
