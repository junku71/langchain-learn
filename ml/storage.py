from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.exists():
        return pd.read_parquet(source)
    csv_path = source.with_suffix(".csv")
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def merge_checkpoint(
    new: pd.DataFrame,
    path: str | Path,
    keys: list[str],
) -> pd.DataFrame:
    existing = read_frame(path)
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new.copy()
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    write_frame(combined, path)
    return combined.reset_index(drop=True)
