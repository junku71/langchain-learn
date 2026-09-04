from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "tickers"
MASTER_BASE_URL = "https://new.real.download.dws.co.kr/common/master"
SUPPORTED_MARKETS = ("KOSPI", "KOSDAQ", "NASDAQ", "SP500")
DOMESTIC_MASTERS = {
    "KOSPI": ("kospi_code.mst.zip", "kospi_code.mst", 228, ".KS"),
    "KOSDAQ": ("kosdaq_code.mst.zip", "kosdaq_code.mst", 222, ".KQ"),
}
OVERSEAS_MASTERS = {
    "NASDAQ": ("nasmst.cod.zip", "nasmst.cod"),
    "NYSE": ("nysmst.cod.zip", "nysmst.cod"),
    "AMEX": ("amsmst.cod.zip", "amsmst.cod"),
}


class TickerMapper:
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, session=None):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self._frames: dict[str, pd.DataFrame] = {}

    @staticmethod
    def normalize_market(market: str) -> str:
        value = market.strip().upper().replace("&", "").replace(" ", "")
        aliases = {
            "KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ", "NASDAQ": "NASDAQ",
            "S&P": "SP500", "S&P500": "SP500", "SP": "SP500", "SP500": "SP500",
        }
        if value not in aliases:
            raise ValueError(f"Unsupported market {market!r}: {SUPPORTED_MARKETS}")
        return aliases[value]

    def csv_path(self, market: str) -> Path:
        return self.cache_dir / f"{self.normalize_market(market).lower()}.csv"

    def _download_zip_member(self, archive: str, member: str) -> bytes:
        response = self.session.get(f"{MASTER_BASE_URL}/{archive}", timeout=30)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zipped:
            members = {name.casefold(): name for name in zipped.namelist()}
            actual_member = members.get(member.casefold())
            if actual_member is None:
                raise ValueError(
                    f"KIS archive {archive} does not contain {member}"
                )
            return zipped.read(actual_member)

    def _download_domestic(self, market: str) -> pd.DataFrame:
        archive, member, tail_width, suffix = DOMESTIC_MASTERS[market]
        content = self._download_zip_member(archive, member)
        records = []
        for line in content.decode("cp949").splitlines():
            prefix = line[:-tail_width]
            if len(prefix) < 22:
                continue
            code, name = prefix[:9].strip(), prefix[21:].strip()
            if code and name:
                records.append({
                    "market": market, "code": code, "name": name,
                    "english_name": "", "ticker": f"{code}{suffix}",
                })
        return pd.DataFrame(records)

    def _download_overseas_exchange(self, exchange: str) -> pd.DataFrame:
        archive, member = OVERSEAS_MASTERS[exchange]
        raw = pd.read_csv(
            BytesIO(self._download_zip_member(archive, member)),
            sep="\t", encoding="cp949", dtype=str,
        )
        if raw.shape[1] < 8:
            raise ValueError(f"Unexpected KIS {exchange} master format")
        result = pd.DataFrame({
            "market": exchange,
            "code": raw.iloc[:, 4].str.strip(),
            "name": raw.iloc[:, 6].fillna("").str.strip(),
            "english_name": raw.iloc[:, 7].fillna("").str.strip(),
            "ticker": raw.iloc[:, 4].str.strip(),
            "index_member": (
                raw.iloc[:, 20].fillna("").str.strip()
                if raw.shape[1] > 20 else ""
            ),
        })
        result["name"] = result["name"].where(
            result["name"].ne(""), result["english_name"]
        )
        return result[result["code"].ne("") & result["name"].ne("")]

    def download_market(self, market: str) -> pd.DataFrame:
        market = self.normalize_market(market)
        if market in DOMESTIC_MASTERS:
            result = self._download_domestic(market)
        elif market == "NASDAQ":
            result = self._download_overseas_exchange("NASDAQ").drop(
                columns="index_member"
            )
        else:
            result = pd.concat(
                [self._download_overseas_exchange(x) for x in ("NASDAQ", "NYSE", "AMEX")],
                ignore_index=True,
            )
            result = result[result["index_member"].eq("1")].copy()
            if result.empty:
                raise NotImplementedError(
                    "KIS US masters do not currently identify S&P 500 members. "
                    f"Provide {self.csv_path('SP500')} with the standard columns."
                )
            result["market"] = "SP500"
            result = result.drop(columns="index_member")
        if result.empty:
            raise ValueError(f"KIS master returned no symbols for {market}")
        result = (
            result.drop_duplicates(subset=["market", "code"])
            .sort_values(["name", "code"]).reset_index(drop=True)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.csv_path(market), index=False, encoding="utf-8-sig")
        self._frames[market] = result
        return result.copy()

    def load_market(self, market: str, refresh: bool = False) -> pd.DataFrame:
        market = self.normalize_market(market)
        if not refresh and market in self._frames:
            return self._frames[market].copy()
        path = self.csv_path(market)
        if path.exists() and not refresh:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            aliases = {
                "종목코드": "code", "단축코드": "code",
                "종목명": "name", "한글명": "name", "한글종목명": "name",
                "영문명": "english_name", "영문종목명": "english_name",
            }
            frame = frame.rename(columns=aliases)
            if not {"code", "name"}.issubset(frame.columns):
                raise ValueError(
                    f"Ticker CSV needs code/name or 종목코드/종목명 columns: {path}"
                )
            frame["market"] = frame.get("market", market)
            frame["english_name"] = frame.get("english_name", "")
            if "ticker" not in frame:
                suffix = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}.get(market, "")
                frame["ticker"] = frame["code"].astype(str) + suffix
            frame = frame[["market", "code", "name", "english_name", "ticker"]]
            self._frames[market] = frame
            return frame.copy()
        return self.download_market(market)

    def load_all(self, refresh: bool = False) -> pd.DataFrame:
        return pd.concat(
            [self.load_market(x, refresh=refresh) for x in SUPPORTED_MARKETS],
            ignore_index=True,
        )

    def _search_frame(self, market: str | None) -> pd.DataFrame:
        if market:
            return self.load_market(market)

        frames = []
        for item in SUPPORTED_MARKETS:
            try:
                frames.append(self.load_market(item))
            except NotImplementedError:
                continue
        if not frames:
            raise KeyError("No ticker master data is available")
        return pd.concat(frames, ignore_index=True)

    def code_to_name(self, code: str, market: str | None = None) -> str:
        if market is None:
            upper_code = code.strip().upper()
            if upper_code.endswith(".KS"):
                market = "KOSPI"
            elif upper_code.endswith(".KQ"):
                market = "KOSDAQ"
        bare = code.strip().split(".", 1)[0].upper()
        frame = self._search_frame(market)
        matches = frame[frame["code"].str.upper().eq(bare)]
        if matches.empty:
            raise KeyError(f"Unknown ticker code: {code}")
        return str(matches.iloc[0]["name"])

    def name_to_code(self, name: str, market: str | None = None) -> str:
        query = name.strip().casefold()
        frame = self._search_frame(market)
        matches = frame[
            frame["name"].str.casefold().eq(query)
            | frame["english_name"].str.casefold().eq(query)
        ]
        if matches.empty:
            raise KeyError(f"Unknown company name: {name}")
        if len(matches["code"].unique()) > 1:
            markets = ", ".join(sorted(matches["market"].unique()))
            raise ValueError(f"Ambiguous name {name!r}; specify market ({markets})")
        return str(matches.iloc[0]["code"])

    def name_to_ticker(self, name: str, market: str | None = None) -> str:
        code = self.name_to_code(name, market)
        frame = self._search_frame(market)
        return str(frame[frame["code"].eq(code)].iloc[0]["ticker"])


_default_mapper = TickerMapper()


def get_company_name(ticker: str, market: str | None = None) -> str:
    try:
        return _default_mapper.code_to_name(ticker, market)
    except (
        KeyError,
        NotImplementedError,
        OSError,
        requests.RequestException,
        ValueError,
    ):
        return ticker.split(".", 1)[0]


def get_ticker_code(company_name: str, market: str | None = None) -> str:
    return _default_mapper.name_to_code(company_name, market)


def get_yfinance_ticker(company_name: str, market: str | None = None) -> str:
    return _default_mapper.name_to_ticker(company_name, market)


def get_domestic_security(company_name: str) -> dict:
    """Resolve an exact company name across KOSPI and KOSDAQ masters."""
    query = company_name.strip().casefold()
    if not query:
        raise KeyError("Company name is empty")
    frame = pd.concat(
        [_default_mapper.load_market("KOSPI"), _default_mapper.load_market("KOSDAQ")],
        ignore_index=True,
    )
    matches = frame[
        frame["name"].str.casefold().eq(query)
        | frame["english_name"].str.casefold().eq(query)
    ].drop_duplicates(subset=["ticker"])
    if matches.empty:
        raise KeyError(f"Unknown domestic company name: {company_name}")
    if len(matches) > 1:
        choices = ", ".join(
            f"{row['market']} {row['ticker']}" for _, row in matches.iterrows()
        )
        raise ValueError(f"Ambiguous company name {company_name!r}: {choices}")
    row = matches.iloc[0]
    return {
        "name": str(row["name"]), "ticker": str(row["ticker"]),
        "market": str(row["market"]),
    }


def get_security(company_name: str, markets: tuple[str, ...]) -> dict:
    """Resolve an exact company name or symbol within selected universes."""
    query = company_name.strip().casefold()
    if not query:
        raise KeyError("Company name is empty")
    frame = pd.concat(
        [_default_mapper.load_market(market) for market in markets],
        ignore_index=True,
    )
    matches = frame[
        frame["name"].str.casefold().eq(query)
        | frame["english_name"].str.casefold().eq(query)
        | frame["code"].str.casefold().eq(query)
        | frame["ticker"].str.casefold().eq(query)
    ].drop_duplicates(subset=["ticker"])
    if matches.empty:
        raise KeyError(f"Unknown security: {company_name}")
    if len(matches) > 1:
        choices = ", ".join(
            f"{row['market']} {row['ticker']}" for _, row in matches.iterrows()
        )
        raise ValueError(f"Ambiguous security {company_name!r}: {choices}")
    row = matches.iloc[0]
    return {
        "name": str(row["name"]), "ticker": str(row["ticker"]),
        "market": str(row["market"]),
    }
