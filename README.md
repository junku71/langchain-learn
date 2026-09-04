# Agentic AI Korean Stock Selection

한국 주식의 기술적 지표, 외국인·기관 수급, DART 재무정보, 시장·거시경제 데이터를 결합해 종목을 선별하는 프로젝트입니다. 핵심은 미래정보 누수를 방지한 point-in-time ML 데이터셋과 LightGBM·Random Forest 앙상블이며, LangGraph 기반 분석 에이전트와 리스크·포트폴리오·모의매매 계층도 포함합니다.

## 주요 기능

- 거래일별 KOSPI·KOSDAQ 시가총액 유니버스 구성
- Yahoo Finance OHLCV 및 글로벌 시장 데이터 수집
- KRX 공식 VKOSPI 수집 및 날짜별 체크포인트 저장
- KIS 외국인·기관 수급 이력 수집
- Open DART 재무제표 기반 point-in-time 재무비율 생성
- 한국은행 ECOS 금리 데이터 수집
- 모멘텀, 기술적 지표, 유동성, 수급, 가치평가 및 횡단면 순위 생성
- Purged walk-forward 방식의 ML 평가
- LightGBM 분류기·랭커와 Random Forest 앙상블
- 시장별 추천 종목 생성
- LangGraph 기반 기술적·기본적·뉴스·수급 분석 흐름
- 모의 브로커, 리스크 엔진, 포트폴리오 제한 및 거래 로그

## ML 유니버스

유니버스는 매 거래일 당시의 시가총액으로 다시 구성합니다.

| 단계 | KOSPI | KOSDAQ | 전체 최대 |
|---|---:|---:|---:|
| 모델 학습 | 시총 상위 100 | 시총 상위 100 | 200 |
| 최신 예측 | 시총 상위 50 | 시총 상위 50 | 100 |
| 최종 추천 | ML 점수 상위 10 | ML 점수 상위 10 | 20 |

`market_cap_rank`는 시장별로 1부터 다시 시작합니다. 과거 데이터도 현재 구성종목을 소급 적용하지 않고 해당 날짜의 KRX 시가총액 자료로 구성하여 survivorship bias를 줄입니다.

주요 설정은 [`ml/config.py`](ml/config.py)에 있습니다.

## 프로젝트 구조

```text
agentic-ai-langchain/
├─ analysis/                     # 시장·기술·재무·뉴스·수급 분석
│  ├─ economy_data.py            # Yahoo/ECOS/KRX VKOSPI 수집과 캐시
│  ├─ technical.py               # RSI, MACD, ATR, ADX, Bollinger 등
│  ├─ fundamental.py             # KIS 기반 현재 재무 분석
│  ├─ news_service.py            # 뉴스·실적 데이터 통합
│  └─ ticker_mapper.py           # 국내외 티커 매핑
├─ broker/                       # 브로커 추상화와 KIS/Paper 구현
├─ ml/                           # ML 데이터·학습·예측 파이프라인
│  ├─ collect_data.py            # 전체 원천 데이터 수집
│  ├─ universe_history.py        # 일별 시장별 시가총액 유니버스
│  ├─ collect_flow_history.py    # KIS 수급 이력
│  ├─ collect_fundamental_history.py # DART 재무 이력
│  ├─ build_dataset.py           # point-in-time feature panel 생성
│  ├─ features.py                # 피처와 학습 타깃 계산
│  ├─ validate_dataset.py        # 구조·커버리지 검증
│  ├─ train_model.py             # walk-forward 평가와 앙상블 학습
│  ├─ predict_top_stocks.py      # 시장별 최신 상위 종목 출력
│  ├─ DATA_SCHEMA.md             # 데이터 소스와 PIT 규칙
│  └─ FEATURE_PANEL_COLUMNS.md   # CSV 46개 컬럼 정의서
├─ paper/                        # 모의매매와 포지션 관리
├─ risk/                         # ATR 기반 리스크 계산
├─ test/                         # pytest 테스트
├─ multiagent_graph.py           # LangGraph 분석·의사결정 그래프
├─ LANGGRAPH_FLOW.md             # 노드·Conditional Edge 상세 리포트
├─ LIVE_TRADING_FLOW.md          # 개장 매수·1분 감시 자동매매 설계
├─ main.py                       # 단일 종목 그래프 실행 예시
├─ portfolio_manager.py          # 종목·섹터·투자비중 제한
├─ pyproject.toml                # Python 버전과 의존성
├─ uv.lock                       # 고정된 의존성 버전
└─ .env.example                  # 환경변수 예시
```

`backup/`은 이전 구현 보관용이며 현재 ML 파이프라인의 실행 진입점이 아닙니다.

## 요구사항 및 설치

- Python 3.13 이상
- [uv](https://docs.astral.sh/uv/)
- 데이터 소스별 API 키

```powershell
uv sync
Copy-Item .env.example .env
```

## 환경 설정

| 환경변수 | 용도 | 필수 시점 |
|---|---|---|
| `OPENAI_API_KEY` | LangGraph의 OpenAI 모델 호출 | 에이전트 실행 |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | 네이버 뉴스 | 뉴스 분석 |
| `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_CANO` | KIS 시세·수급·재무 및 실거래 연결 | KIS 데이터/브로커 사용 |
| `KIS_ACCOUNT_TYPE` | `VIRTUAL` 또는 `REAL` | KIS 사용 |
| `KIS_ENABLE_TRADING` | 실제 주문 허용 여부 | KIS 브로커 사용 |
| `ECOS_API_KEY` | 한국은행 금리 데이터 | ML 데이터 수집 |
| `DART_API_KEY` | DART 재무제표 | 재무 이력 수집 |
| `KRX_API_KEY` | 일별 시가총액 유니버스와 VKOSPI | ML 데이터 수집 |
| `BROKER_TYPE` | `paper` 또는 `kis` | 에이전트/매매 실행 |

처음에는 안전한 설정을 권장합니다.

```dotenv
BROKER_TYPE=paper
KIS_ACCOUNT_TYPE=VIRTUAL
KIS_ENABLE_TRADING=false
```

KRX 키에는 KOSPI·KOSDAQ 일별매매정보와 파생상품지수 시세정보(VKOSPI) 승인이 필요합니다.

## ML 빠른 시작

### 1. API 연결 확인

```powershell
uv run python -m ml.check_krx_api
uv run python -m ml.check_data_sources --ticker 005930.KS --days 45
```

### 2. 전체 학습 데이터 준비

```powershell
uv run python -m ml.prepare_training_data --years 5
```

최종 데이터셋은 `2021-10-01`부터 저장됩니다. 그 이전 원천 가격은 20일·60일 rolling 지표의 lookback으로 활용할 수 있으므로 캐시에서 즉시 삭제하지 않습니다.

수급 데이터를 이미 받았다면:

```powershell
uv run python -m ml.prepare_training_data --years 5 --skip-flow
```

수집이 끝났고 feature panel만 다시 만들고 싶다면:

```powershell
uv run python -m ml.prepare_training_data --skip-collect
```

### 3. 모델 학습

```powershell
uv run python -m ml.train_model
```

터미널에는 walk-forward fold, 개별 모델 학습 단계, 경과시간과 예상 남은 시간이 출력됩니다.

### 4. 시장별 상위 종목 예측

```powershell
uv run python -m ml.predict_top_stocks --top 10
```

`--top 10`은 전체 10개가 아니라 시장별 10개를 뜻합니다. 정상적인 경우 KOSPI 10개와 KOSDAQ 10개, 최대 20개가 출력됩니다.

## 단계별 ML 실행

```powershell
# 1. 유니버스, 가격, 수급, 재무, 경제 데이터
uv run python -m ml.collect_data --start 2021-08-01

# 2. feature panel 생성
uv run python -m ml.build_dataset

# 3. 데이터 품질 검증
uv run python -m ml.validate_dataset

# 4. 앙상블 학습
uv run python -m ml.train_model

# 5. 시장별 추천
uv run python -m ml.predict_top_stocks --top 10
```

빠른 API smoke test는 한 거래일만 수집하고 수급·재무를 생략합니다.

```powershell
uv run python -m ml.collect_data --start 2025-01-02 --limit-sessions 1 --skip-flow --skip-fundamental
```

## 수급과 재무 데이터만 수집

```powershell
uv run python -m ml.collect_flow_history --years 5
uv run python -m ml.collect_fundamental_history --years 5
```

KIS 수급 이력 API는 평일 장중 제한될 수 있습니다. 장 마감 이후 또는 주말에 실행하는 것이 안정적이며, 수집기는 기존 CSV를 체크포인트로 사용해 이어받습니다.

과거 시가총액 유니버스에 등장한 모든 종목의 수급을 보강하려면:

```powershell
uv run python -c "from ml.storage import read_frame; from ml.config import UNIVERSE_HISTORY_PATH; from ml.collect_flow_history import collect_flow_history; u=read_frame(UNIVERSE_HISTORY_PATH); collect_flow_history(years=5, tickers=sorted(u['ticker'].dropna().unique()))"
```

## 주요 산출물

| 파일 | 설명 |
|---|---|
| `data/ml/raw/universe_history.parquet` | 거래일별 KOSPI·KOSDAQ 시장별 시가총액 유니버스 |
| `data/ml/raw/price_history.parquet` | 연속 OHLCV 가격 이력 |
| `data/ml/flow_history.csv` | 외국인·기관 순매수 이력 |
| `data/ml/fundamental_history.csv` | DART 공시일 기준 재무 이력 |
| `data/ml/universe_top200.csv` | 현재 시장별 상위 종목과 섹터 매핑 |
| `data/ml/kis_sector_cache.csv` | KIS 섹터 조회 캐시 |
| `data/economy/economic_indicators.csv` | 통합 시장·거시경제 캐시 |
| `data/economy/vkospi.csv` | KRX VKOSPI 날짜별 체크포인트 |
| `data/ml/processed/feature_panel.parquet` | 모델 학습용 기준 데이터셋 |
| `data/ml/processed/feature_panel.csv` | 사람이 확인하기 쉬운 46개 컬럼 CSV |
| `data/ml/reports/data_quality.json` | 데이터 구조·피처 커버리지 검증 결과 |
| `data/ml/reports/model_report.json` | 모델 설정과 성능지표 |
| `data/ml/reports/feature_importance.csv` | 모델별 및 가중 피처 중요도 |
| `data/ml/reports/walk_forward_predictions.parquet` | Out-of-sample walk-forward 예측 |
| `data/ml/reports/top_selection.csv` | 최신 시장별 상위 10개 추천 |
| `ml/models/korea_top200_ensemble.joblib` | 학습 모델·메타데이터·최신 예측 묶음 |

컬럼별 계산식과 의미는 [`ml/FEATURE_PANEL_COLUMNS.md`](ml/FEATURE_PANEL_COLUMNS.md), 데이터 공개시점과 미래정보 누수 방지 규칙은 [`ml/DATA_SCHEMA.md`](ml/DATA_SCHEMA.md)를 참고하세요.

## LangGraph 분석 실행

`main.py`는 시장 데이터에서 기술·기본·뉴스·수급 분석을 병렬 수행한 뒤 ML 필터, 리스크, 포트폴리오 제한, 최종 의사결정과 브로커 실행으로 이어지는 예제입니다.

전체 노드 그래프, Conditional Edge와 State 입출력은 [`LANGGRAPH_FLOW.md`](LANGGRAPH_FLOW.md)를 참고하세요.

매일 개장 시 신규 매수하고 장중 1분마다 손절·익절을 감시하는 운영형 시스템 제안은 [`LIVE_TRADING_FLOW.md`](LIVE_TRADING_FLOW.md)를 참고하세요.

먼저 학습 모델을 만든 뒤 `main.py`의 `initial_state`에서 `ticker`, `sector`, `account_size`, `risk_per_trade`, `trailing_stop_pct`를 실행 목적에 맞게 설정합니다.

```powershell
uv run python main.py
```

실거래 브로커를 사용할 때는 `.env`의 `BROKER_TYPE=kis`와 실전용 KIS 자격정보가 필요합니다. `KIS_ENABLE_TRADING=true`는 실제 주문을 허용하므로 충분히 검증하기 전에는 사용하지 마세요.

## 테스트

```powershell
# 전체 테스트
uv run pytest -q

# 영역별 테스트
uv run pytest test/test_ml_redesign.py -q
uv run pytest test/test_economy_data.py -q
uv run pytest test/test_ml_filter.py -q
```

## 데이터 설계 원칙

- 유니버스는 매 거래일 당시 시가총액으로 구성합니다.
- 재무정보는 DART 공개일 이후에만 backward-as-of로 연결합니다.
- 오늘의 재무정보나 현재 구성종목을 과거 행에 backfill하지 않습니다.
- 미국시장·환율·원자재 관측치는 한국 거래일 기준으로 지연해 사용합니다.
- rolling 피처와 미래 타깃은 연속 가격 이력에서 먼저 계산한 뒤 해당 날짜 유니버스로 제한합니다.
- 최종 feature panel은 `2021-10-01` 이후만 저장하되 계산용 lookback 원천 데이터는 유지합니다.
- 수집하지 못한 값과 실제 0을 같은 의미로 처리하지 않습니다.

## 문제 해결

### KRX 401 오류

```powershell
uv run python -m ml.check_krx_api
```

KRX 마이페이지에서 KOSPI·KOSDAQ 일별매매정보와 파생상품지수 시세정보가 현재 키에 승인됐는지 확인합니다.

### VKOSPI가 비어 있는 경우

VKOSPI는 Yahoo의 `^VKOSPI`가 아니라 KRX `drvprod_dd_trd` API에서 받습니다.

```powershell
uv run python -c "import pandas as pd; d=pd.read_csv('data/economy/vkospi.csv'); print(d.tail()); print(d['VKOSPI'].notna().sum())"
```

### 수급 이력이 중단되는 경우

KIS 일별 수급 이력은 장중 제한될 수 있습니다. 기존 체크포인트는 유지되므로 장 마감 후 같은 명령을 다시 실행합니다.

### 최종 데이터셋만 다시 만들기

```powershell
uv run python -m ml.prepare_training_data --skip-collect
```

### 모델이 오래된 데이터로 학습된 경우

데이터셋이나 유니버스 정의가 바뀌면 모델을 다시 학습합니다.

```powershell
uv run python -m ml.train_model
```

## 주의사항

이 프로젝트는 연구·교육 및 전략 검증 목적입니다. 모델 점수와 에이전트 출력은 수익을 보장하지 않으며 실제 투자 판단을 대신하지 않습니다. 수수료, 세금, 슬리피지, 체결 가능성, 데이터 지연과 API 장애를 별도로 고려해야 합니다.

## 일일 자동매매 서비스

[`LIVE_TRADING_FLOW.md`](LIVE_TRADING_FLOW.md)의 장전 준비, 개장 신규 매수, 장중 1분 감시, 장 마감 정산을 `trading/` 패키지로 구현했습니다. 기본값은 `TRADING_DRY_RUN=true`이며 주문을 제출하지 않습니다.

```powershell
# 현재 시각에 실행할 작업을 한 번만 판정·실행
uv run python -m trading.cli once

# KIS/PaperBroker 잔고와 현재 보유종목 조회
uv run python -m trading.cli status

# 기존 JSON 형식이 필요한 경우
uv run python -m trading.cli status --json

# 특정 시각을 재현해 스케줄 판정 확인
uv run python -m trading.cli once --at "2026-08-31T09:05:00+09:00"

# 계속 실행하며 거래 세션에 맞춰 작업 예약
uv run python -m trading.cli run
```

### 번호 선택형 거래 콘솔

잔고·보유종목·보호가격·주문 이력 조회, 사용자 지정 매수/매도, 보호가격 변경 및 Scheduler 제어는 다음 콘솔에서 실행합니다.

```powershell
uv run python -m trading.console
```

### Streamlit Web GUI

브라우저 기반 운영 화면은 다음 명령으로 실행합니다.

```powershell
uv run streamlit run trading/web_app.py
```

기본 주소는 `http://localhost:8501`입니다. 대시보드, 보유종목·보호가격, 주문내역과
미체결 취소, API reconciliation, 수동 매수·매도, Scheduler·Kill Switch·ML Filter,
Top10, LLM 리밸런싱 승인/Override와 감사 로그를 제어할 수 있습니다.

실제 주문에는 화면에 표시되는 확인문구가 필요합니다. Streamlit Scheduler를 사용할
때는 별도 터미널의 `trading.cli run`을 동시에 실행하지 마십시오. 웹서버는 기본
localhost로만 운영하고 외부 공개 시 별도 인증·TLS·방화벽을 구성해야 합니다.

2번 사용자 지정 매수는 종목명을 입력받아 KOSPI·KOSDAQ 종목코드와 섹터를 자동으로 찾습니다. 별도의 섹터 입력은 받지 않습니다. 매수가격은 시장가, 지정가, 최우선지정가(가격우선), 최유리지정가(타이밍우선)를 지원합니다. 정규장에는 선택한 방식으로 즉시 주문하고, 장 시작 전에는 당일 개장 예약, 장 마감 후·주말·휴일에는 다음 영업일 예약 Queue에 저장합니다. 수동 주문은 Kill Switch, 신규 매수 상태, 일일 주문 수와 포트폴리오 비중을 검사하고 SQLite 주문 의도·감사 로그에 기록합니다. REAL 계좌에서는 `BUY 종목코드` 또는 `SELL 종목코드` 확인 문구를 정확히 입력해야 주문됩니다.

3번 `사용자 지정 종목 매도`는 보유종목 선택표에 평균단가와 `평균단가 × 수량` 총평가금액을 함께 표시합니다. 매도 방식은 시장가, 지정가, 최우선지정가(가격우선), 최유리지정가(타이밍우선)를 지원하며 KIS 주문구분 코드로 전달됩니다.

KIS 체결·미체결 조회와 메뉴 4의 미체결 조회 및 잔량 전량 취소를 지원합니다. 취소 접수 후 최종 `CANCELLED` 상태는 reconciliation에서 확인합니다. 콘솔의 미체결 화면은 마지막 reconciliation 결과이므로 API 장애나 조회 주기 사이에는 KIS 화면과 차이가 있을 수 있습니다. 별도 터미널에서 `trading.cli run`을 이미 실행 중이라면 콘솔 안의 Scheduler를 중복 시작하지 마십시오.

KIS 체결 reconciliation은 정규장 중 매분 주문번호를 조회해 `SUBMITTED`, `PARTIALLY_FILLED`, `FILLED`, `REJECTED`, `CANCELLED` 상태를 SQLite에 반영합니다. Slack 알림은 Incoming Webhook 또는 Bot Token 방식으로 설정할 수 있습니다.

```dotenv
SLACK_NOTIFICATIONS_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Bot Token을 사용하면 `SLACK_BOT_TOKEN` 또는 `SLACK_API_KEY`와 `SLACK_CHANNEL`을 설정합니다. 연결 확인은 거래 콘솔의 `17. Slack 알림 연결 테스트`에서 수행합니다. 주문 취소는 메뉴 4에서 할 수 있으며 현재는 미체결 잔량 전량 취소만 지원합니다.

거래 콘솔의 `10. 오늘의 Top10 pick`은 먼저 `KOSPI`, `KOSDAQ`, `Both` 중 분석 유니버스를 선택하고, 후보군에 기술 30%, 기본 25%, 뉴스·실적 20%, 수급 25% 점수를 적용해 상위 10개와 요소별 추천근거를 표로 표시합니다. 분석 후 기본·기술·수급·뉴스의 점수와 상세 근거를 담은 PDF를 OneDrive Public 리포트 폴더에 저장하고 Slack으로 링크를 전송합니다. `Both`는 양 시장 후보를 합쳐 전체 Top10을 선정합니다. ML Filter ON은 장전 ML snapshot을 사용합니다. OFF는 학습 모델을 호출하지 않고 `universe_top200.csv`의 선택 시장을 분석하며, `Both`에서는 KOSPI 시가총액 상위 100개와 KOSDAQ 시가총액 상위 100개, 총 200개를 분석합니다. 추천은 읽기 전용이며 자동 주문을 발생시키지 않습니다. 거래일·전략 버전·후보소스·유니버스 크기·시장 선택별로 SQLite에 캐시되고 `REFRESH` 입력 시 해당 시장을 다시 분석합니다. 완성된 분석 결과는 SQLite의 `top_recommendations_latest`에 저장되어 11번 리밸런싱 입력으로 재사용됩니다.

거래 콘솔의 자동매매 `6. 손절·익절·Trailing 변경`은 서브메뉴에서 개별 종목 변경 또는 전 종목 일괄 적용을 선택합니다. 일괄 적용은 평균단가에서 `-3×일봉 ATR`을 손절가로, 평균단가의 `+20%`를 익절가로 계산하고 기본 8% trailing stop을 함께 적용합니다.

주요 환경변수는 `.env.example`의 `TRADING_*` 항목을 참고합니다. 실행 상태, job 중복 방지 키, 주문 의도와 손절·익절 상태는 기본적으로 `data/trading/live_trading.sqlite3`에 저장됩니다.

현재 브로커 인터페이스는 체결 조회와 1분 OHLC를 제공하지 않으므로 KIS 연결 시에는 현재가를 OHLC가 모두 같은 1분 스냅샷으로 취급합니다. 따라서 실계좌 전환 전 KIS 분봉 공급자와 주문 체결 reconciliation을 추가하고, 공식 KRX 휴장일·단축장 캘린더를 주입해야 합니다. 먼저 PaperBroker와 `TRADING_DRY_RUN=true`로 검증하십시오.

```powershell
uv run python -m pytest test/test_live_trading.py -q
```

### KIS 1분봉 매도 감시와 체결 reconciliation

`BROKER_TYPE=kis`이면 장중 매도 감시기는 KIS 국내주식 당일 분봉 API의 직전 완료
1분봉을 사용합니다. 따라서 스케줄러 호출 시점의 현재가뿐 아니라 해당 분봉의 고가와
저가도 손절가, 익절가, trailing stop 판정에 반영됩니다. 분봉 API 조회가 실패하면
기본적으로 KIS 현재가로 감시를 계속합니다.

```dotenv
TRADING_MONITOR_INTERVAL_SECONDS=60
TRADING_KIS_MINUTE_BARS_ENABLED=true
TRADING_KIS_MINUTE_FALLBACK_TO_QUOTE=true
TRADING_KIS_COMPLETED_BARS_ONLY=true
```

정규장 중 매 분마다 체결 reconciliation을 먼저 실행한 뒤 보유종목 매도 감시를
실행합니다. reconciliation은 KIS 주문번호로 체결내역을 조회하여 `SUBMITTED`,
`PARTIALLY_FILLED`, `FILLED`, `REJECTED`, `CANCELLED` 상태를 SQLite에 반영하고,
상태 변경 알림을 Slack으로 한 번만 전송합니다. 콘솔 메뉴의 수동 실행에서
`5. reconciliation`을 선택해 즉시 확인할 수도 있습니다.

### LLM 포트폴리오 리밸런싱

거래 콘솔의 11번 메뉴는 보유종목 수익률·비중, 10번에서 마지막으로 완료한 당일
Top10과 시장뉴스를 함께 검토해 구조화된 리밸런싱 제안을 생성합니다. 10번에서 선택한
`KOSPI`, `KOSDAQ`, `Both` 범위와 결과를 그대로 사용하며 다른 시장을 자동 재분석하지
않습니다. 당일 10번 결과가 없으면 11번은 실행되지 않습니다. 기본값은 비활성화이며 다음 설정을
명시한 경우에만 사용할 수 있습니다.

```dotenv
TRADING_REBALANCE_ENABLED=true
TRADING_REBALANCE_LLM_MODEL=gpt-5.6
TRADING_REBALANCE_MAX_TURNOVER_PCT=0.30
TRADING_REBALANCE_MAX_POSITION_PCT=0.20
TRADING_REBALANCE_MAX_SECTOR_PCT=0.40
TRADING_REBALANCE_MIN_CASH_PCT=0.10
TRADING_REBALANCE_MIN_CONFIDENCE=0.70
TRADING_REBALANCE_PROPOSAL_TTL_MINUTES=10
TRADING_REBALANCE_FILL_WAIT_SECONDS=30
TRADING_REBALANCE_REPORT_DIR=C:\Users\junku\OneDrive\Public\AI-Stock-Agent
TRADING_REBALANCE_REPORT_BASE_URL=https://1drv.ms/f/c/714ca3cac310f853/IgCM0r1KM0xSQb1g5kThUnk-AdzbHMoYLJZwWYglmBPg4qM?e=FM3eQu
```

LLM은 목표 비중만 제안합니다. Risk Validator가 최초 주문수량을 계산한 뒤 사용자는
각 종목에 대해 승인, BUY/SELL 및 수량 수정, 제외 또는 전체 취소를 선택해야 합니다.
검토된 주문은 보유수량, Top10 외 신규매수, 종목·섹터 비중, 현금과 회전율 정책을
다시 검증합니다. 종목별 검토가 완료되지 않은 제안서는 실행할 수 없으며, 이후에도
제안서별 최종 확인 문구가 필요합니다. 매도 미체결 상태에서는 후속 매수를 실행하지 않습니다.

현금비중 또는 회전율 정책으로 거부된 제안은 `OVERRIDE <제안서 ID>` 확인문구로
사용자가 위험을 명시적으로 수락할 수 있습니다. Override는 감사 로그와 Slack에
기록됩니다. Top10 외 신규매수, 시세 누락, 중복 제안과 종목·섹터 비중 위반은
주문 무결성 오류로 취급하여 Override할 수 없습니다.

리밸런싱 입력에는 시장뉴스뿐 아니라 보유종목과 Top10 추천종목별 최신 뉴스 분석도
포함됩니다. 제안 근거는 OneDrive의 `Public/AI-Stock-Agent` 아래
`Rebalancing_Proposal_YYYYMMDD.pdf`로 저장됩니다. Slack에는 해당 파일명과 OneDrive
공개 폴더 링크를 전송하므로 Bot Token과 Webhook 방식 모두 사용할 수 있습니다.

매수가 포함된 리밸런싱은 실행 전에 `Kill Switch=NORMAL`, `Buy=ON`을 요구합니다.
정규장 외 시간에 최종 승인된 리밸런싱 BUY/SELL 주문은 즉시 제출하지 않고 SQLite
예약 Queue에 저장됩니다. 장 시작 전 승인은 당일, 장 마감 후·주말·휴일 승인은 다음
영업일에 실행되며 Scheduler가 정규장 개시 후 SELL 주문부터 한 번만 제출합니다.
예약 내역은 콘솔 `1. Buy/Sell 예약종목 조회 및 취소/정정`에서 확인할 수 있으며,
아직 실행되지 않은 예약은 취소하거나 주문 수량을 정정할 수 있습니다.
실행 도중 중단된 제안을 재시도하면 현재 보유수량과 목표수량의 차이만 주문하여 이미
체결된 매도를 중복 제출하지 않습니다.
체결 대기는 현재 제안에 포함된 매도 종목만 대상으로 하므로 BUY-only 리밸런싱은
무관한 과거 미체결 기록 때문에 차단되지 않습니다.
