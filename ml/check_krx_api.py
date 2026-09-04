from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import requests
from dotenv import load_dotenv


ENDPOINTS = {
    "KOSPI daily trading": "stk_bydd_trd",
    "KOSDAQ daily trading": "ksq_bydd_trd",
}


def _previous_weekday(value: date) -> date:
    value -= timedelta(days=1)
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def check_krx_api(
    historical_date: str = "20250102",
    session: requests.Session | None = None,
) -> list[dict]:
    load_dotenv()
    api_key = os.getenv("KRX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("KRX_API_KEY is missing from .env")
    client = session or requests.Session()
    dates = {
        "recent": _previous_weekday(date.today()).strftime("%Y%m%d"),
        "historical": historical_date.replace("-", ""),
    }
    results = []
    for service, endpoint in ENDPOINTS.items():
        for period, base_date in dates.items():
            response = client.get(
                f"https://data-dbg.krx.co.kr/svc/apis/sto/{endpoint}",
                params={"basDd": base_date},
                headers={"AUTH_KEY": api_key},
                timeout=20,
            )
            try:
                payload = response.json()
            except requests.exceptions.JSONDecodeError:
                payload = {}
            rows = payload.get("OutBlock_1") or []
            results.append({
                "service": service,
                "endpoint": endpoint,
                "period": period,
                "date": base_date,
                "http_status": response.status_code,
                "response_code": payload.get("respCode"),
                "response_message": payload.get("respMsg"),
                "rows": len(rows),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose KRX Open API authorization")
    parser.add_argument("--historical-date", default="2025-01-02")
    args = parser.parse_args()
    results = check_krx_api(args.historical_date)
    for result in results:
        print(
            f"{result['service']} [{result['period']} {result['date']}]: "
            f"HTTP {result['http_status']}, rows={result['rows']}, "
            f"code={result['response_code']}, message={result['response_message']}"
        )
    if all(result["http_status"] == 401 for result in results):
        print(
            "All calls were unauthorized. In KRX My Page > API usage status, "
            "verify that both daily-trading products are approved for this key."
        )


if __name__ == "__main__":
    main()
