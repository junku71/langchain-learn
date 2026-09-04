import json
import hashlib
import os
import re
import time
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv

from broker.base import Broker
from broker.models import OrderExecution, OrderResult, Position


class KISAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class KISConfig:
    app_key: str
    app_secret: str
    cano: str
    account_product_code: str = "01"
    account_type: Literal["VIRTUAL", "REAL"] = "VIRTUAL"
    enable_trading: bool = False
    token_file: str | None = None
    timeout: float = 10.0

    def __post_init__(self):
        normalized = self.account_type.strip().upper()

        if normalized not in {"VIRTUAL", "REAL"}:
            raise ValueError(
                "account_type must be 'VIRTUAL' or 'REAL'"
            )

        object.__setattr__(self, "account_type", normalized)

    @classmethod
    def from_env(cls) -> "KISConfig":
        load_dotenv()
        required = {
            "KIS_APP_KEY": os.getenv("KIS_APP_KEY", ""),
            "KIS_APP_SECRET": os.getenv("KIS_APP_SECRET", ""),
            "KIS_CANO": os.getenv("KIS_CANO", ""),
        }
        missing = [key for key, value in required.items() if not value]

        if missing:
            raise ValueError(
                f"Missing KIS configuration: {', '.join(missing)}"
            )

        account_type = os.getenv(
            "KIS_ACCOUNT_TYPE",
            "VIRTUAL",
        ).strip().upper()

        return cls(
            app_key=required["KIS_APP_KEY"],
            app_secret=required["KIS_APP_SECRET"],
            cano=required["KIS_CANO"],
            account_product_code=os.getenv(
                "KIS_ACNT_PRDT_CD",
                "01",
            ),
            account_type=account_type,
            enable_trading=(
                os.getenv("KIS_ENABLE_TRADING", "false")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            token_file=os.getenv("KIS_TOKEN_FILE") or None,
        )


class KISBroker(Broker):
    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    VIRTUAL_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(
        self,
        config: KISConfig,
        session: requests.Session | None = None,
    ):
        self.config = config
        self.session = session or requests.Session()
        self.base_url = (
            self.REAL_BASE_URL
            if config.account_type == "REAL"
            else self.VIRTUAL_BASE_URL
        )
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self._position_metadata: dict[str, dict] = {}
        self._balance_cache: dict | None = None
        self._balance_cache_at = 0.0
        self._last_request_at = 0.0
        self._load_cached_token()

    def _token_cache_key(self) -> str:
        identity = (
            f"{self.config.account_type}:"
            f"{self.config.app_key}"
        )
        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_env(cls) -> "KISBroker":
        return cls(KISConfig.from_env())

    @staticmethod
    def _stock_code(ticker: str) -> str:
        code = ticker.split(".", 1)[0].upper()

        if not re.fullmatch(r"[0-9A-Z]{6}", code):
            raise ValueError(
                f"KIS domestic ticker must contain a 6-character code: {ticker}"
            )

        return code

    @staticmethod
    def _as_float(value) -> float:
        if value in (None, ""):
            return 0.0
        return float(str(value).replace(",", ""))

    def _load_cached_token(self) -> None:
        if not self.config.token_file:
            return

        path = Path(self.config.token_file)

        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            expires_at = float(data["expires_at"])

            if (
                data.get("cache_key") == self._token_cache_key()
                and expires_at > time.time() + 60
            ):
                self._access_token = data["access_token"]
                self._token_expires_at = expires_at
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _save_cached_token(self) -> None:
        if not self.config.token_file or not self._access_token:
            return

        path = Path(self.config.token_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "access_token": self._access_token,
                "expires_at": self._token_expires_at,
                "cache_key": self._token_cache_key(),
            }),
            encoding="utf-8",
        )

    def _authenticate(self) -> str:
        if (
            self._access_token
            and self._token_expires_at > time.time() + 60
        ):
            return self._access_token

        response = self.session.post(
            f"{self.base_url}/oauth2/tokenP",
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if "access_token" not in data:
            raise KISAPIError(
                data.get("error_description", "KIS token issuance failed")
            )

        self._access_token = data["access_token"]
        self._token_expires_at = (
            time.time() + int(data.get("expires_in", 86_400))
        )
        self._save_cached_token()
        return self._access_token

    def _headers(self, tr_id: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self._authenticate()}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _invalidate_access_token(self) -> None:
        """Discard a token rejected by KIS even if its local expiry is in the future."""
        self._access_token = None
        self._token_expires_at = 0.0

    def _request(
        self,
        method: str,
        path: str,
        tr_id: str,
        *,
        params: dict | None = None,
        body: dict | None = None,
    ) -> dict:
        min_interval = (
            0.55
            if self.config.account_type == "VIRTUAL"
            else 0.05
        )

        for attempt in range(2):
            elapsed = time.monotonic() - self._last_request_at

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(tr_id),
                params=params,
                json=body,
                timeout=self.config.timeout,
            )
            self._last_request_at = time.monotonic()

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError as error:
                raise KISAPIError(
                    f"KIS HTTP {response.status_code} {path}: "
                    "invalid JSON response"
                ) from error

            message_code = data.get("msg_cd")

            if message_code == "EGW00123" and attempt == 0:
                # KIS can invalidate a token earlier than the locally cached
                # expires_in value. Force issuance of a new token and retry once.
                self._invalidate_access_token()
                continue

            if message_code == "EGW00201" and attempt == 0:
                time.sleep(1.0)
                continue

            if not response.ok:
                raise KISAPIError(
                    f"KIS HTTP {response.status_code} {path} - "
                    f"{message_code or 'HTTP_ERROR'}: "
                    f"{data.get('msg1', data.get('error_description', 'Request failed'))}"
                )

            if data.get("rt_cd") != "0":
                raise KISAPIError(
                    f"KIS {path} - "
                    f"{message_code or 'KIS_ERROR'}: "
                    f"{data.get('msg1', 'KIS request failed')}"
                )

            return data

        raise KISAPIError(f"KIS {path}: request retry exhausted")

    def get_current_price(self, ticker: str) -> float:
        data = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": self._stock_code(ticker),
            },
        )
        price = self._as_float(data["output"]["stck_prpr"])

        if price <= 0:
            raise KISAPIError(f"Invalid current price for {ticker}")

        return price

    def get_minute_bars(
        self,
        ticker: str,
        at: datetime,
    ) -> list[dict]:
        """Return normalized KIS intraday minute bars ending near ``at``.

        KIS returns at most 120 rows from the requested date/time.  The
        provider layer decides whether to use the current (possibly forming)
        bar or the most recent completed bar.
        """
        local_at = at
        data = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
            "FHKST03010230",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": self._stock_code(ticker),
                "FID_INPUT_HOUR_1": local_at.strftime("%H%M%S"),
                "FID_INPUT_DATE_1": local_at.strftime("%Y%m%d"),
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "",
            },
        )
        output = data.get("output2") or []
        if isinstance(output, dict):
            output = [output]

        bars: list[dict] = []
        for row in output:
            raw_date = str(row.get("stck_bsop_date") or "")
            raw_time = str(row.get("stck_cntg_hour") or "").zfill(6)
            try:
                timestamp = datetime.strptime(
                    f"{raw_date}{raw_time}", "%Y%m%d%H%M%S"
                ).replace(tzinfo=at.tzinfo)
                values = {
                    "open": self._as_float(row.get("stck_oprc")),
                    "high": self._as_float(row.get("stck_hgpr")),
                    "low": self._as_float(row.get("stck_lwpr")),
                    "close": self._as_float(row.get("stck_prpr")),
                }
            except (TypeError, ValueError):
                continue
            if min(values.values()) <= 0:
                continue
            bars.append({
                "ticker": ticker,
                "timestamp": timestamp,
                **values,
                "volume": int(self._as_float(row.get("cntg_vol"))),
            })

        return sorted(bars, key=lambda bar: bar["timestamp"])

    def get_stock_sector(self, ticker: str) -> dict:
        code = self._stock_code(ticker)
        output = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
            },
        ).get("output") or {}
        sector = str(output.get("bstp_kor_isnm") or "").strip()
        market = str(output.get("rprs_mrkt_kor_name") or "").strip()
        return {
            "code": code,
            "kis_sector": sector or "UNKNOWN",
            "kis_market": market or None,
        }

    def get_fundamental_data(self, ticker: str) -> dict:
        code = self._stock_code(ticker)
        common_params = {
            "FID_INPUT_ISCD": code,
            "FID_DIV_CLS_CODE": "0",
            "FID_COND_MRKT_DIV_CODE": "J",
        }
        quote = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
            },
        ).get("output") or {}
        profitability = self._request(
            "GET",
            "/uapi/domestic-stock/v1/finance/profit-ratio",
            "FHKST66430400",
            params=common_params,
        ).get("output") or []
        stability = self._request(
            "GET",
            "/uapi/domestic-stock/v1/finance/stability-ratio",
            "FHKST66430600",
            params=common_params,
        ).get("output") or []
        other_ratios = self._request(
            "GET",
            "/uapi/domestic-stock/v1/finance/other-major-ratios",
            "FHKST66430500",
            params=common_params,
        ).get("output") or []
        growth = self._request(
            "GET",
            "/uapi/domestic-stock/v1/finance/growth-ratio",
            "FHKST66430800",
            params=common_params,
        ).get("output") or []

        return {
            "ticker": ticker,
            "PER": self._optional_float(quote.get("per")),
            "PBR": self._optional_float(quote.get("pbr")),
            "ROE": self._latest_ratio(
                profitability,
                "self_cptl_ntin_inrt",
            ),
            "debt_ratio": self._latest_ratio(stability, "lblt_rate"),
            # The current KIS quote response does not expose PCR. Preserve
            # missing as None instead of treating unavailable data as zero.
            "PCR": self._optional_float(quote.get("pcr")),
            "EV_EBITDA": self._latest_ratio(other_ratios, "ev_ebitda"),
            "revenue_growth": self._latest_ratio(growth, "grs"),
            "operating_profit_growth": self._latest_ratio(
                growth,
                "bsop_prfi_inrt",
            ),
        }

    def get_investor_flow(self, ticker: str) -> list[dict]:
        data = self._request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            "FHKST01010900",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": self._stock_code(ticker),
            },
        )
        output = data.get("output") or []
        return output if isinstance(output, list) else [output]

    def get_investor_flow_history(
        self,
        ticker: str,
        start: date | datetime,
        end: date | datetime,
    ) -> list[dict]:
        """Return daily foreign/institution net quantities for a date range."""
        start_date = start.date() if isinstance(start, datetime) else start
        cursor = end.date() if isinstance(end, datetime) else end
        while cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
        rows: dict[str, dict] = {}

        while cursor >= start_date:
            data = self._request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
                "FHPTJ04160001",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": self._stock_code(ticker),
                    "FID_INPUT_DATE_1": cursor.strftime("%Y%m%d"),
                    "FID_ORG_ADJ_PRC": "",
                    "FID_ETC_CLS_CODE": "",
                },
            )
            output = data.get("output2") or []
            if isinstance(output, dict):
                output = [output]

            returned_dates = []
            for item in output:
                raw_date = str(item.get("stck_bsop_date", ""))
                try:
                    trading_date = datetime.strptime(raw_date, "%Y%m%d").date()
                except ValueError:
                    continue
                returned_dates.append(trading_date)
                if start_date <= trading_date <= cursor:
                    rows[raw_date] = {
                        "date": trading_date.isoformat(),
                        "ticker": ticker,
                        "foreign_net": self._as_float(item.get("frgn_ntby_qty")),
                        "institution_net": self._as_float(item.get("orgn_ntby_qty")),
                    }

            if not returned_dates:
                break
            next_cursor = min(returned_dates) - timedelta(days=1)
            if next_cursor >= cursor:
                break
            cursor = next_cursor

        return sorted(rows.values(), key=lambda row: row["date"])

    @staticmethod
    def _optional_float(value) -> float | None:
        if value in (None, "", "-"):
            return None

        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _latest_ratio(
        cls,
        rows: list[dict] | dict,
        field: str,
    ) -> float | None:
        if isinstance(rows, dict):
            rows = [rows]

        dated_rows = sorted(
            rows,
            key=lambda row: str(row.get("stac_yymm", "")),
            reverse=True,
        )

        for row in dated_rows:
            value = cls._optional_float(row.get(field))

            if value is not None:
                return value

        return None

    def _balance_data(self) -> dict:
        if (
            self._balance_cache is not None
            and time.monotonic() - self._balance_cache_at < 1.0
        ):
            return self._balance_cache

        tr_id = (
            "TTTC8434R"
            if self.config.account_type == "REAL"
            else "VTTC8434R"
        )
        data = self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id,
            params={
                "CANO": self.config.cano,
                "ACNT_PRDT_CD": self.config.account_product_code,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        self._balance_cache = data
        self._balance_cache_at = time.monotonic()
        return data

    def get_balance(self) -> dict:
        data = self._balance_data()
        summary = (data.get("output2") or [{}])[0]

        return {
            "initial_cash": 0.0,
            "cash": self._as_float(summary.get("dnca_tot_amt")),
            "d2_cash": self._as_float(summary.get("prvs_rcdl_excc_amt")),
            "realized_pnl": 0.0,
            "total_commission": 0.0,
            "position_count": len(data.get("output1") or []),
            "stock_market_value": self._as_float(summary.get("scts_evlu_amt")),
            "stock_purchase_amount": self._as_float(
                summary.get("pchs_amt_smtl_amt")
            ),
            "total_equity": self._as_float(
                summary.get("tot_evlu_amt")
            ),
            "unrealized_pnl": self._as_float(
                summary.get("evlu_pfls_smtl_amt")
            ),
        }

    def get_positions(self) -> dict[str, Position]:
        data = self._balance_data()
        positions = {}

        for item in data.get("output1") or []:
            quantity = int(self._as_float(item.get("hldg_qty")))

            if quantity <= 0:
                continue

            code = item["pdno"]
            ticker = f"{code}.KS"
            metadata = self._position_metadata.get(code, {})
            positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=self._as_float(item.get("pchs_avg_pric")),
                sector=metadata.get("sector"),
                stop_loss=metadata.get("stop_loss"),
                take_profit=metadata.get("take_profit"),
                trailing_stop_pct=metadata.get("trailing_stop_pct"),
                trailing_stop=metadata.get("trailing_stop"),
                highest_price=metadata.get("highest_price"),
            )

        return positions

    def get_position(self, ticker: str) -> Position | None:
        code = self._stock_code(ticker)
        return self.get_positions().get(f"{code}.KS")

    def get_order_execution(
        self,
        order_id: str,
        order_date: str,
        ticker: str | None = None,
    ) -> OrderExecution | None:
        """Query KIS daily order/execution state for one broker order number."""
        compact_date = order_date.replace("-", "")
        if len(compact_date) != 8 or not compact_date.isdigit():
            raise ValueError(f"Invalid order date: {order_date}")
        tr_id = "TTTC0081R" if self.config.account_type == "REAL" else "VTTC0081R"
        data = self._request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id,
            params={
                "CANO": self.config.cano,
                "ACNT_PRDT_CD": self.config.account_product_code,
                "INQR_STRT_DT": compact_date,
                "INQR_END_DT": compact_date,
                "SLL_BUY_DVSN_CD": "00",
                "PDNO": self._stock_code(ticker) if ticker else "",
                "CCLD_DVSN": "00",
                "INQR_DVSN": "00",
                "INQR_DVSN_3": "00",
                "ORD_GNO_BRNO": "",
                "ODNO": str(order_id),
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
                "EXCG_ID_DVSN_CD": "KRX",
            },
        )
        rows = data.get("output1") or []
        if isinstance(rows, dict):
            rows = [rows]
        matches = [row for row in rows if str(row.get("odno", "")).lstrip("0") == str(order_id).lstrip("0")]
        if not matches:
            return None
        row = matches[0]
        ordered = int(self._as_float(row.get("ord_qty")))
        filled = int(self._as_float(row.get("tot_ccld_qty") or row.get("ccld_qty")))
        cancelled_quantity = int(self._as_float(row.get("cncl_cfrm_qty")))
        raw_remaining = row.get("rmn_qty")
        if raw_remaining in (None, ""):
            remaining = max(ordered - filled - cancelled_quantity, 0)
        else:
            # An explicit zero is authoritative. Reconstructing ordered-filled
            # here resurrects orders already cancelled by KIS virtual trading.
            remaining = int(self._as_float(raw_remaining))
        rejected = str(row.get("rjct_yn", "")).upper() == "Y"
        cancelled = (
            str(row.get("cncl_yn", "")).upper() == "Y"
            or (cancelled_quantity > 0 and remaining == 0)
        )
        if rejected:
            status = "REJECTED"
        elif ordered > 0 and filled >= ordered:
            status = "FILLED"
            remaining = 0
        elif cancelled:
            status = "CANCELLED"
        elif filled > 0:
            status = "PARTIALLY_FILLED"
        else:
            status = "SUBMITTED"
        code = str(row.get("pdno") or (self._stock_code(ticker) if ticker else ""))
        average_fill_price = self._as_float(
            row.get("avg_prvs")
            or row.get("avg_ccld_unpr")
            or row.get("ccld_unpr")
        )
        if average_fill_price <= 0 and filled > 0:
            total_fill_amount = self._as_float(
                row.get("tot_ccld_amt") or row.get("ccld_amt")
            )
            if total_fill_amount > 0:
                average_fill_price = total_fill_amount / filled
        return OrderExecution(
            order_id=str(order_id),
            ticker=ticker or (f"{code}.KS" if code else ""),
            side="BUY" if str(row.get("sll_buy_dvsn_cd", "")) == "02" else "SELL",
            status=status,
            ordered_quantity=ordered,
            filled_quantity=filled,
            remaining_quantity=remaining,
            order_price=self._as_float(row.get("ord_unpr")),
            average_fill_price=average_fill_price,
            order_date=str(row.get("ord_dt") or compact_date),
            order_time=str(row.get("ord_tmd") or ""),
            name=str(row.get("prdt_name") or ""),
            raw=row,
        )

    def list_order_executions(self, order_date: str) -> list[OrderExecution]:
        """Return the KIS daily order ledger, including orders absent locally."""
        compact_date = order_date.replace("-", "")
        if len(compact_date) != 8 or not compact_date.isdigit():
            raise ValueError(f"Invalid order date: {order_date}")
        tr_id = "TTTC0081R" if self.config.account_type == "REAL" else "VTTC0081R"
        data = self._request(
            "GET", "/uapi/domestic-stock/v1/trading/inquire-daily-ccld", tr_id,
            params={
                "CANO": self.config.cano,
                "ACNT_PRDT_CD": self.config.account_product_code,
                "INQR_STRT_DT": compact_date,
                "INQR_END_DT": compact_date,
                "SLL_BUY_DVSN_CD": "00",
                "PDNO": "",
                "CCLD_DVSN": "00",
                "INQR_DVSN": "00",
                "INQR_DVSN_3": "00",
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
                "EXCG_ID_DVSN_CD": "KRX",
            },
        )
        rows = data.get("output1") or []
        if isinstance(rows, dict):
            rows = [rows]
        executions = []
        for row in rows:
            order_id = str(row.get("odno") or "").strip()
            code = str(row.get("pdno") or "").strip()
            if not order_id or not code:
                continue
            ordered = int(self._as_float(row.get("ord_qty")))
            filled = int(self._as_float(row.get("tot_ccld_qty") or row.get("ccld_qty")))
            cancelled_quantity = int(self._as_float(row.get("cncl_cfrm_qty")))
            raw_remaining = row.get("rmn_qty")
            remaining = (
                max(ordered - filled - cancelled_quantity, 0)
                if raw_remaining in (None, "")
                else int(self._as_float(raw_remaining))
            )
            rejected = str(row.get("rjct_yn", "")).upper() == "Y"
            cancelled = str(row.get("cncl_yn", "")).upper() == "Y" or (
                cancelled_quantity > 0 and remaining == 0
            )
            if rejected:
                status = "REJECTED"
            elif ordered > 0 and filled >= ordered:
                status, remaining = "FILLED", 0
            elif cancelled:
                status = "CANCELLED"
            elif filled > 0:
                status = "PARTIALLY_FILLED"
            else:
                status = "SUBMITTED"
            average_fill_price = self._as_float(
                row.get("avg_prvs") or row.get("avg_ccld_unpr") or row.get("ccld_unpr")
            )
            if average_fill_price <= 0 and filled > 0:
                fill_amount = self._as_float(row.get("tot_ccld_amt") or row.get("ccld_amt"))
                if fill_amount > 0:
                    average_fill_price = fill_amount / filled
            executions.append(OrderExecution(
                order_id=order_id,
                ticker=f"{code}.KS",
                side="BUY" if str(row.get("sll_buy_dvsn_cd", "")) == "02" else "SELL",
                status=status,
                ordered_quantity=ordered,
                filled_quantity=filled,
                remaining_quantity=remaining,
                order_price=self._as_float(row.get("ord_unpr")),
                average_fill_price=average_fill_price,
                order_date=str(row.get("ord_dt") or compact_date),
                order_time=str(row.get("ord_tmd") or ""),
                name=str(row.get("prdt_name") or ""),
                raw=row,
            ))
        return executions

    def cancel_order(
        self,
        order_id: str,
        order_date: str,
        ticker: str | None = None,
    ) -> OrderResult:
        """Cancel the entire remaining quantity of a domestic stock order."""
        if not self.config.enable_trading:
            return OrderResult(
                status="REJECTED", ticker=ticker or "", side="CANCEL",
                reason="KIS trading is disabled",
            )
        execution = self.get_order_execution(order_id, order_date, ticker)
        if execution is None:
            return OrderResult(
                status="REJECTED", ticker=ticker or "", side="CANCEL",
                reason="KIS original order was not found",
            )
        if execution.status not in {"SUBMITTED", "PARTIALLY_FILLED"}:
            return OrderResult(
                status="REJECTED", ticker=execution.ticker, side="CANCEL",
                quantity=execution.remaining_quantity,
                reason=f"Order is not cancellable: {execution.status}",
            )
        if execution.remaining_quantity <= 0:
            return OrderResult(
                status="REJECTED", ticker=execution.ticker, side="CANCEL",
                reason="No remaining quantity to cancel",
            )
        raw = execution.raw
        organization = str(
            raw.get("ord_gno_brno")
            or raw.get("krx_fwdg_ord_orgno")
            or raw.get("ord_orgno")
            or ""
        ).strip()
        if not organization:
            return OrderResult(
                status="REJECTED", ticker=execution.ticker, side="CANCEL",
                reason="KIS order organization number is unavailable",
            )
        tr_id = "TTTC0013U" if self.config.account_type == "REAL" else "VTTC0013U"
        try:
            data = self._request(
                "POST",
                "/uapi/domestic-stock/v1/trading/order-rvsecncl",
                tr_id,
                body={
                    "CANO": self.config.cano,
                    "ACNT_PRDT_CD": self.config.account_product_code,
                    "KRX_FWDG_ORD_ORGNO": organization,
                    "ORGN_ODNO": str(order_id),
                    "ORD_DVSN": str(raw.get("ord_dvsn_cd") or "00"),
                    "RVSE_CNCL_DVSN_CD": "02",
                    "ORD_QTY": "0",
                    "ORD_UNPR": "0",
                    "QTY_ALL_ORD_YN": "Y",
                    "EXCG_ID_DVSN_CD": "KRX",
                    "CNDT_PRIC": "",
                },
            )
        except (KISAPIError, requests.RequestException, ValueError) as error:
            return OrderResult(
                status="REJECTED", ticker=execution.ticker, side="CANCEL",
                quantity=execution.remaining_quantity, reason=str(error),
            )
        output = data.get("output") or {}
        return OrderResult(
            status="CANCEL_SUBMITTED",
            ticker=execution.ticker,
            side="CANCEL",
            quantity=execution.remaining_quantity,
            order_id=str(output.get("ODNO") or output.get("odno") or ""),
            reason=f"Cancel requested for original order {order_id}",
        )

    def _order(
        self,
        side: str,
        ticker: str,
        price: float,
        quantity: int,
        reason: str,
        order_type: str = "LIMIT",
    ) -> OrderResult:
        if not self.config.enable_trading:
            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side=side,
                price=price,
                quantity=quantity,
                reason="KIS trading is disabled",
            )

        if price <= 0 or quantity <= 0:
            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side=side,
                price=price,
                quantity=quantity,
                reason="Invalid price or quantity",
            )

        is_buy = side == "BUY"
        tr_id = {
            ("REAL", True): "TTTC0802U",
            ("REAL", False): "TTTC0801U",
            ("VIRTUAL", True): "VTTC0802U",
            ("VIRTUAL", False): "VTTC0801U",
        }[(self.config.account_type, is_buy)]
        order_codes = {
            "LIMIT": "00", "MARKET": "01",
            "BEST_LIMIT": "03", "PRIORITY_LIMIT": "04",
        }
        normalized_order_type = order_type.strip().upper()
        if normalized_order_type not in order_codes:
            return OrderResult(
                status="REJECTED", ticker=ticker, side=side, price=price,
                quantity=quantity, reason=f"Unsupported order type: {order_type}",
            )
        order_code = order_codes[normalized_order_type]
        order_price = str(int(price)) if normalized_order_type == "LIMIT" else "0"

        try:
            data = self._request(
                "POST",
                "/uapi/domestic-stock/v1/trading/order-cash",
                tr_id,
                body={
                    "CANO": self.config.cano,
                    "ACNT_PRDT_CD": self.config.account_product_code,
                    "PDNO": self._stock_code(ticker),
                    "ORD_DVSN": order_code,
                    "ORD_QTY": str(quantity),
                    "ORD_UNPR": order_price,
                    "EXCG_ID_DVSN_CD": "KRX",
                    "SLL_TYPE": "01" if not is_buy else "",
                    "CNDT_PRIC": "",
                },
            )
        except (KISAPIError, requests.RequestException, ValueError) as error:
            return OrderResult(
                status="REJECTED",
                ticker=ticker,
                side=side,
                price=price,
                quantity=quantity,
                reason=str(error),
            )

        output = data.get("output") or {}
        return OrderResult(
            status="SUBMITTED",
            ticker=ticker,
            side=side,
            price=price,
            quantity=quantity,
            order_id=str(output.get("ODNO", "")),
            reason=reason,
        )

    def buy(
        self,
        ticker: str,
        price: float,
        quantity: int,
        sector: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> OrderResult:
        result = self._order(
            "BUY",
            ticker,
            price,
            quantity,
            reason,
            order_type,
        )

        if result.status == "SUBMITTED":
            code = self._stock_code(ticker)
            self._position_metadata[code] = {
                "sector": sector,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop_pct": trailing_stop_pct,
                "trailing_stop": (
                    price * (1 - trailing_stop_pct)
                    if trailing_stop_pct is not None
                    else None
                ),
                "highest_price": price,
            }

        return result

    def sell(
        self,
        ticker: str,
        price: float,
        quantity: int,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> OrderResult:
        return self._order(
            "SELL",
            ticker,
            price,
            quantity,
            reason,
            order_type,
        )
