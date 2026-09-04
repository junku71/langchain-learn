from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


DEFAULT_CACHE_PATH = Path("data/economy/economic_indicators.csv")
VKOSPI_CACHE_PATH = Path("data/economy/vkospi.csv")
KRX_DERIVATIVE_INDEX_URL = (
    "https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd"
)

# Continuous market series. KIS exposes several index/FX/futures endpoints, but
# it does not expose one continuous-history API for every requested asset. These
# Yahoo symbols are isolated behind a provider so they can be replaced by KIS.
MARKET_SERIES = {
    # Equity indices
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    # VKOSPI is collected from the official KRX derivative-index API below.
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "SOX": "^SOX",
    "NIKKEI225": "^N225",
    "HANG_SENG": "^HSI",
    "SHANGHAI": "000001.SS",
    "EURO_STOXX50": "^STOXX50E",
    # Major futures
    "SP500_FUTURES": "ES=F",
    "NASDAQ100_FUTURES": "NQ=F",
    "DOW_FUTURES": "YM=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "NATURAL_GAS": "NG=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F",
    "ALUMINUM": "ALI=F",
    # Exchange-traded commodity proxies where continuous futures are absent.
    "LEAD": "LEED.L",
    "ZINC": "ZINC.L",
    "NICKEL": "NICK.L",
    # KRW crosses
    "USD_KRW": "KRW=X",
    "JPY_KRW": "JPYKRW=X",
    "EUR_KRW": "EURKRW=X",
    # Yahoo's direct CNY/KRW history is empty; invert its liquid KRW/CNY pair.
    "CNY_KRW": "KRWCNY=X",
}
INVERTED_MARKET_SERIES = {"CNY_KRW"}

# ECOS statistic/item codes can be overridden without changing application code.
ECOS_SERIES = {
    "CALL_RATE": ("817Y002", "010101000"),
    "CD91_RATE": ("817Y002", "010502000"),
    "KTB3Y_RATE": ("817Y002", "010200000"),
    "CORP_BOND3Y_RATE": ("817Y002", "010300000"),
}

# LME lead/zinc/nickel/tin continuous histories are not consistently available
# through KIS/Yahoo. Keeping the columns explicit preserves the feature schema
# and missing values retain the meaning "no observation" rather than zero.
RATE_COLUMNS = list(ECOS_SERIES)
ECONOMIC_LEVEL_COLUMNS = [
    *MARKET_SERIES,
    "VKOSPI",
    *RATE_COLUMNS,
]


class EconomicDataProvider(Protocol):
    def fetch(self, start: date, end: date) -> pd.DataFrame: ...


@dataclass
class KrxVkospiProvider:
    """Collect the official KOSPI 200 volatility index with a daily checkpoint."""

    api_key: str
    cache_path: str | Path = VKOSPI_CACHE_PATH
    session: requests.Session | None = None
    timeout: float = 30.0
    checkpoint_every: int = 20
    request_interval: float = 0.05

    @classmethod
    def from_env(cls) -> KrxVkospiProvider | None:
        load_dotenv()
        api_key = os.getenv("KRX_API_KEY", "").strip()
        return cls(api_key) if api_key else None

    @staticmethod
    def _is_vkospi(row: dict) -> bool:
        name = (
            str(row.get("IDX_NM", ""))
            .upper()
            .replace(" ", "")
            .replace("-", "")
        )
        return "VKOSPI" in name or ("코스피200" in name and "변동성" in name)

    def _load_cache(self) -> pd.DataFrame:
        path = Path(self.cache_path)
        if not path.exists():
            return pd.DataFrame(
                columns=["VKOSPI"],
                index=pd.DatetimeIndex([], name="Date"),
            )
        cached = pd.read_csv(path, index_col="Date", parse_dates=True)
        if "VKOSPI" not in cached:
            cached["VKOSPI"] = float("nan")
        cached.index = pd.to_datetime(cached.index).tz_localize(None).normalize()
        return cached[["VKOSPI"]].loc[
            lambda frame: ~frame.index.duplicated(keep="last")
        ].sort_index()

    def _save_cache(self, frame: pd.DataFrame) -> None:
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        frame.sort_index().rename_axis("Date").to_csv(
            temporary,
            encoding="utf-8-sig",
        )
        temporary.replace(path)

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        cached = self._load_cache()
        requested = pd.bdate_range(start, end).normalize()
        missing = requested.difference(cached.index)
        client = self.session or requests.Session()
        additions: list[dict] = []

        for number, session_date in enumerate(missing, start=1):
            response = client.get(
                KRX_DERIVATIVE_INDEX_URL,
                params={"basDd": session_date.strftime("%Y%m%d")},
                headers={"AUTH_KEY": self.api_key},
                timeout=self.timeout,
            )
            if response.status_code == 401:
                raise RuntimeError(
                    "KRX rejected the derivative-index request (401). Approve "
                    "'파생상품지수 시세정보' for the configured KRX_API_KEY."
                )
            response.raise_for_status()
            rows = response.json().get("OutBlock_1", [])
            match = next((row for row in rows if self._is_vkospi(row)), None)
            raw_value = match.get("CLSPRC_IDX") if match else None
            value = pd.to_numeric(
                str(raw_value).replace(",", "") if raw_value is not None else None,
                errors="coerce",
            )
            # Cache empty weekdays too: public holidays must not be retried forever.
            additions.append({"Date": session_date, "VKOSPI": value})

            if len(additions) >= self.checkpoint_every or number == len(missing):
                addition = pd.DataFrame(additions).set_index("Date")
                cached = pd.concat([cached, addition])
                cached = cached.loc[
                    ~cached.index.duplicated(keep="last")
                ].sort_index()
                cached["VKOSPI"] = pd.to_numeric(cached["VKOSPI"], errors="coerce")
                self._save_cache(cached)
                additions = []
            if self.request_interval > 0 and number < len(missing):
                time.sleep(self.request_interval)

        return cached.reindex(requested)


@dataclass
class YahooMarketProvider:
    series: dict[str, str] | None = None

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        symbols = self.series or MARKET_SERIES
        if start > end:
            return pd.DataFrame()

        raw = yf.download(
            list(symbols.values()),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=next(iter(symbols.values())))
        result = pd.DataFrame(index=pd.to_datetime(close.index).tz_localize(None))
        for name, symbol in symbols.items():
            if symbol in close.columns:
                values = pd.to_numeric(close[symbol], errors="coerce")
                result[name] = 1 / values if name in INVERTED_MARKET_SERIES else values
        return result


@dataclass
class EcosRateProvider:
    api_key: str
    session: requests.Session | None = None
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> EcosRateProvider | None:
        load_dotenv()
        api_key = os.getenv("ECOS_API_KEY", "").strip()
        return cls(api_key) if api_key else None

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        session = self.session or requests.Session()
        result = pd.DataFrame()
        for name, (stat_code, item_code) in ECOS_SERIES.items():
            url = (
                "https://ecos.bok.or.kr/api/StatisticSearch/"
                f"{self.api_key}/json/kr/1/100000/{stat_code}/D/"
                f"{start:%Y%m%d}/{end:%Y%m%d}/{item_code}"
            )
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("StatisticSearch", {}).get("row", [])
            series = {
                pd.to_datetime(row["TIME"]): pd.to_numeric(
                    row.get("DATA_VALUE"), errors="coerce"
                )
                for row in rows
                if row.get("TIME")
            }
            result[name] = pd.Series(series, dtype="float64")
        return result.sort_index()


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.apply(pd.to_numeric, errors="coerce")


def _fetch_all(
    start: date,
    end: date,
    providers: list[EconomicDataProvider],
) -> pd.DataFrame:
    frames = [_normalize(provider.fetch(start, end)) for provider in providers]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, axis=1).loc[:, lambda df: ~df.columns.duplicated(keep="last")] if frames else pd.DataFrame()


def _merge_prefer_new(existing: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = _normalize(existing)
    for frame in frames:
        normalized = _normalize(frame)
        if not normalized.empty:
            combined = normalized.combine_first(combined)
    return _normalize(combined)


def update_economy_data(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    providers: list[EconomicDataProvider] | None = None,
) -> pd.DataFrame:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end or date.today()).date()
    if start_date > end_date:
        raise ValueError("start must not be after end")

    path = Path(cache_path)
    existing = pd.DataFrame()
    if path.exists():
        existing = _normalize(pd.read_csv(path, index_col="Date", parse_dates=True))
    schema_changed = set(existing.columns) != set(ECONOMIC_LEVEL_COLUMNS)

    active_providers = providers
    yahoo_provider = None
    ecos_provider = None
    vkospi_provider = None
    if active_providers is None:
        yahoo_provider = YahooMarketProvider()
        ecos_provider = EcosRateProvider.from_env()
        active_providers = [yahoo_provider]
        vkospi_provider = KrxVkospiProvider.from_env()
        if vkospi_provider:
            active_providers.append(vkospi_provider)
        if ecos_provider:
            active_providers.append(ecos_provider)

    fetched = []
    if existing.empty:
        fetched.append(_fetch_all(start_date, end_date, active_providers))
    else:
        first, last = existing.index.min().date(), existing.index.max().date()
        if start_date < first:
            fetched.append(_fetch_all(start_date, first - timedelta(days=1), active_providers))
        if end_date > last:
            fetched.append(_fetch_all(last + timedelta(days=1), end_date, active_providers))

        requested = existing.loc[str(start_date):str(end_date)]
        if providers is None:
            missing_market = {
                name: symbol
                for name, symbol in MARKET_SERIES.items()
                if name not in requested
                or requested[name].notna().sum() < max(2, len(requested) * 0.2)
            }
            if missing_market:
                fetched.append(
                    YahooMarketProvider(missing_market).fetch(start_date, end_date)
                )
            if (
                vkospi_provider
                and (
                    "VKOSPI" not in requested
                    or not requested["VKOSPI"].notna().any()
                )
            ):
                fetched.append(vkospi_provider.fetch(start_date, end_date))
            missing_rates = [
                name
                for name in RATE_COLUMNS
                if name not in requested or not requested[name].notna().any()
            ]
            if missing_rates and ecos_provider:
                fetched.append(ecos_provider.fetch(start_date, end_date))

    combined = _merge_prefer_new(existing, fetched)
    for column in ECONOMIC_LEVEL_COLUMNS:
        if column not in combined:
            combined[column] = float("nan")
    combined = combined[ECONOMIC_LEVEL_COLUMNS]

    if not path.exists() or fetched or schema_changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            combined.rename_axis("Date").to_csv(
                temporary_path,
                encoding="utf-8-sig",
            )
            temporary_path.replace(path)
        except PermissionError:
            temporary_path.unlink(missing_ok=True)
            warnings.warn(
                f"Economic cache is locked and was not updated: {path}",
                RuntimeWarning,
                stacklevel=2,
            )
    return combined.loc[str(start_date):str(end_date)]


def add_economic_features(
    stock_df: pd.DataFrame,
    economy_df: pd.DataFrame | None = None,
    *,
    update: bool = True,
) -> pd.DataFrame:
    if stock_df.empty:
        return stock_df.copy()

    result = stock_df.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    if economy_df is None:
        if update:
            economy_df = update_economy_data(result.index.min(), result.index.max())
        elif DEFAULT_CACHE_PATH.exists():
            economy_df = pd.read_csv(DEFAULT_CACHE_PATH, index_col="Date", parse_dates=True)
        else:
            economy_df = pd.DataFrame(index=result.index)

    levels = preprocess_economy_data(economy_df, result.index)
    for column in ECONOMIC_LEVEL_COLUMNS:
        values = levels[column] if column in levels else pd.Series(index=result.index, dtype=float)
        if column in RATE_COLUMNS:
            result[f"ECON_{column}_CHANGE_1D"] = values.diff(1)
            result[f"ECON_{column}_CHANGE_5D"] = values.diff(5)
        else:
            result[f"ECON_{column}_RETURN_1D"] = values.pct_change(1, fill_method=None)
            result[f"ECON_{column}_RETURN_5D"] = values.pct_change(5, fill_method=None)
    return result


def preprocess_economy_data(
    economy_df: pd.DataFrame,
    target_index: pd.Index,
) -> pd.DataFrame:
    """Align to stock sessions and fill only from already-known observations."""
    normalized = _normalize(economy_df)
    index = pd.to_datetime(target_index).tz_localize(None).normalize()
    if normalized.empty:
        return pd.DataFrame(index=index)

    # Reindex on the union first so observations made on a foreign-market day
    # are available to the next domestic session. Never backfill: that would
    # leak a future observation into training.
    union_index = normalized.index.union(index).sort_values()
    aligned = normalized.reindex(union_index)
    result = pd.DataFrame(index=union_index)
    for column in aligned:
        # Market observations should not survive a long data outage. Rates are
        # slower-moving and may legitimately remain unchanged for several weeks.
        limit = 30 if column in RATE_COLUMNS else 5
        result[column] = aligned[column].ffill(limit=limit)
    return result.reindex(index)


ECONOMIC_FEATURE_COLUMNS = [
    feature
    for column in ECONOMIC_LEVEL_COLUMNS
    for feature in (
        (f"ECON_{column}_CHANGE_1D", f"ECON_{column}_CHANGE_5D")
        if column in RATE_COLUMNS
        else (f"ECON_{column}_RETURN_1D", f"ECON_{column}_RETURN_5D")
    )
]
