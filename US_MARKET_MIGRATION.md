# 미국시장 모드

기존 한국시장 모드는 기본값으로 유지하며, 아래 설정으로 NASDAQ/S&P 500
분석 및 Paper Trading 모드를 활성화한다.

```dotenv
TRADING_MARKET_REGION=US
BROKER_TYPE=paper
PAPER_INITIAL_CASH=100000
TRADING_DRY_RUN=true
TRADING_ML_FILTER_ENABLED=false
```

미국 모드는 `America/New_York` 시간대와 정규장 09:30~16:00을 사용하므로
한국의 서머타임 변환을 코드 밖에서 계산할 필요가 없다. Top10 메뉴의 선택지는
NASDAQ/S&P500으로 바뀌고, yfinance 심볼(예: `AAPL`, `MSFT`)을 그대로 사용한다.

## 데이터와 분석

- 종목 마스터: KIS 해외종목 마스터를 `data/tickers/nasdaq.csv`와
  `data/tickers/sp500.csv`에 UTF-8 CSV로 캐시한다.
- S&P500 멤버 플래그가 원본 마스터에 없으면 표준 컬럼
  (`market,code,name,english_name,ticker`)의 `sp500.csv`를 제공해야 한다.
- 기술/뉴스 분석은 기존 파이프라인을 재사용한다.
- 기본 분석은 Yahoo 지표, 수급 분석은 미국시장에 맞는 20일 가격 모멘텀과
  최근 거래량 참여도로 대체한다. 한국의 외국인/기관 순매수는 사용하지 않는다.
- 한국시장용 ML 후보 모델은 미국 모드에서 자동 비활성화된다. 미국용 모델을
  별도로 학습하기 전까지 종목 마스터 기반 후보를 사용한다.

## 주문 안전성

현재 `KISBroker`는 국내주식 주문/잔고 endpoint 구현이다. 미국 모드에서
`BROKER_TYPE=kis`를 지정하면 잘못된 국내 주문을 막기 위해 시작 단계에서
오류를 낸다. Paper Trading으로 시간대, 예약주문, 리밸런싱을 검증한 뒤
KIS 해외주식 또는 다른 미국 브로커 adapter를 연결해야 실주문이 가능하다.

휴장일과 조기폐장일은 브로커/거래소의 공식 캘린더를 서비스 시작 시
`UsMarketCalendar.holidays`와 `special_sessions`에 주입해야 한다.
