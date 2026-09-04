# Streamlit Trading GUI

## 실행

```powershell
uv run streamlit run trading/web_app.py
```

브라우저에서 `http://localhost:8501`로 접속한다.

## 화면 구성

| 화면 | 주요 기능 |
|---|---|
| 대시보드 | 예수금, 총평가금액, 평가손익, 보유종목 현황 |
| 포트폴리오 | 수익률·보호가격 조회, 종목별 및 ATR 일괄 보호가격 설정 |
| 주문·체결 | 주문 조회, KIS reconciliation, 미체결 잔량 취소 |
| 수동 주문 | 지정가 매수·매도와 보호가격 설정 |
| 자동매매 | Scheduler, 신규 매수, ML Filter, Kill Switch, Slack, 수동 작업 |
| Top10·리밸런싱 | 종합분석, LLM 제안서, OneDrive 리포트, 승인·Override |
| 감사 로그 | 주요 제어·주문·리밸런싱 이벤트 조회 |

## 운영 안전

실제 주문과 위험 제어에는 화면에 제시된 확인 문구를 정확히 입력해야 한다. 브라우저를
외부 네트워크에 공개하지 말고, 필요한 경우 Reverse Proxy에서 인증·TLS·IP 제한을
적용한다. 기존 `uv run python -m trading.cli run` Scheduler와 Streamlit Scheduler를
동시에 실행하지 않는다.
