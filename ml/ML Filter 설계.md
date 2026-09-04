

## Machin Learning Target 
	10영업일 뒤 상승확률 제일 높은 top 10 종목을 복수의 머신러닝알고리즘 사용하여 **Ranking** 으로 
	Training Universe에서 매일 골라내는 것
	입력 데이터는 
	

## 알고리즘 

총 3개 모델
```
	Stock Selection Ranker=LGBM, 
	Main=LGBM
	Base=Random Forest
```
	
| 모델                  | 종류     | Target                               | 역할            |   비중  |
| --------------------- |   ------ | -------------------------------     | --------------- | ------- |
| ① Random Forest       | **분류** | 10일 후 상승/초과수익 여부            | Baseline        |  0.2    |
| ② LightGBM            | **분류** | 10일 후 상승/초과수익 여부            | Main prediction |  0.3    |
| ③ LightGBM Ranker     | **랭킹** | 종목 간 미래(10영업일뒤)수익률 순위    | Stock Selection |  0.5    |

	
# 동작방식
	10영업일 Stock Selection ML Filter에는 100~200종목 → 하나의 Global Model → 매일 종목별 Rank 방식이 더 적합하
	
	
## Feature 개수
	개수를 증가시키는 것보다 20-35개의 강한 Feature선택이 중요.
	아래는 Feature의 그룹별 예시이고 중요도를 얘기한다.
	
| 그룹             | 핵심 feature                   |   중요도 |
| --------------- | ------------------------------ | ----:   |
| Momentum        | RET 5/20/60, relative momentum | ★★★★★ |
| Trend           | MA20/60, ADX, RSI, BB, MACD    |  ★★★★ |
| Volatility      | ATR, realized vol, gap         | ★★★★★ |
| Volume          | volume ratio, turnover         |  ★★★★ |
| 외국인           | 5/20D normalized net buy       | ★★★★★ |
| 기관              5/20D normalized net buy       |  ★★★★ |
| Sector          | sector relative strength       | ★★★★★ |
| Market          | KOSPI trend, VKOSPI            | ★★★★★ |
| Macro           | USD/KRW, SOX/Nasdaq            |  ★★★★ |
| Fundamental     | ROE, PBR, earnings growth      |   ★★★ |
| Cross-sectional | momentum/flow/value rank       | ★★★★★ |

## Training Universe(100) & Prediction Universe(50) 정의

Point-in-Time Universe
```
2018-01-02 → 그날의 Top 100
2018-01-03 → 그날의 Top 100
...
2024-01-02 → 그날의 Top 100
...
2026-08-30 → 그날의 Top 100
```


```
DATE        CODE     MktCapRank   InUniverse
2025-01-02  005930       1          1
2025-01-02  000660       2          1
...
2025-01-02  123456      99          1
2025-01-02  654321     101          0
```

Training Universe는 100종목, Prediction은 Top 50에 대해서만.

## ML 입력 및 출력 데이터 스키마 

### 예시

아래는유니버스가 100종목이고 Feature가 10개라고 가정했을 때 데이터의 모습을 예시한 것임.

| Date | Stock  |  PER | PBR |  ROE | RSI | ADX | ATR% | RET20 |   외인수급 | SectorRS |
| ---- | ------ | ---: | --: | ---: | --: | --: | ---: | ----: | -----: | -------: |
| 8/1  | 삼성전자   | 18.2 | 1.5 |  9.1 |  63 |  25 |  2.1 |  5.2% |  0.32% |     0.71 |
| 8/1  | SK하이닉스 | 11.3 | 2.4 | 22.1 |  71 |  32 |  3.4 | 12.1% |  0.81% |     0.94 |
| 8/1  | 현대차    |  6.2 | 0.7 | 13.4 |  42 |  18 |  2.3 | -2.1% | -0.12% |     0.43 |
| 8/1  | 한화에어로  | 25.3 | 5.2 | 19.8 |  68 |  37 |  4.1 | 15.3% |  0.47% |     0.89 |
| 8/2  | 삼성전자   | 18.5 | 1.6 |  9.1 |  65 |  26 |  2.2 |  5.8% |  0.41% |     0.73 |
| 8/2  | SK하이닉스 | 11.7 | 2.5 | 22.1 |  74 |  34 |  3.6 | 13.4% |  0.92% |     0.96 |
| ...  | ...    |  ... | ... |  ... | ... | ... |  ... |   ... |    ... |      ... |

이경우 10년치 (2500거래일) 데이터를 모은다고 가정했을 때
100개 종목 × 2,500 거래일이면:
$$ 100 \times 2500 = 250,000 rows $$
가 됩니다.
Feature가 50개라면 ML 입력 행렬 X는 대략:
$$ X = 250,000 \times 50 $$


### 입출력 변수 스키마 제안
INPUT X
────────────────────────────
① Momentum (4)
   RET5, RET20, RET60, RET20_SECTOR_RELATIVE

② Technical (5)
   RSI, ADX, ATR%, MACD,Bollinger 

③ Volume(3)
   VolumeRatio, Turnover,liquidity

④ Flow(4)
   ForeignFlow5/20 Ratio normalized to MarketCap
   InstitutionFlow5/20 Ratio normalized to MarketCap

⑤ Fundamental(6)
   ROE
   PBR, PBR sector rank
   PER, PER sector rank
   EV/EBITDA rank

⑦ Market(8)
   KOSPI 지수
   NASDAQ 지수
   S&P 지수
   VKOSPI
   USDKRW
   EURKRW
   금현물
   유가

⑧ Cross-sectional(3)
   Momentum rank
   Flow rank
   Value rank
   
   
────────────────────────────
           ↓
    Global LightGBM(비중0.3) + Random Forest(비중0.2) + LightGBM Ranker(비중0.5)
           ↓

출력의 모습
OUTPUT
────────────────────────────
Stock         ML Score ML Rank
SK하이닉스       0.87   1
한화에어로       0.84   2
삼성전자         0.79   3
...
────────────────────────────
           ↓
       TOP 10~20



