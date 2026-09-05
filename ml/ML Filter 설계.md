

# ML Filter 설계(확정 코드 기준)

## 1. 목표

현재 코드 기준의 ML Filter는 다음 목표를 가진다.

- 매일 거래일마다 point-in-time 데이터 기반으로 종목별 점수를 산출한다.
- 같은 날짜의 종목들 사이에서 상대순위(`return_rank`)를 계산한다.
- 최종적으로 `ml_score`를 기준으로 상위 종목을 선택한다.
- 생산 환경에서는 Top 10 종목을 추천 후보로 사용한다.

핵심 목적은 단순히 “상승 확률을 예측”하는 것만이 아니라,
“같은 날짜의 종목들 중 어디가 상대적으로 우수한가”를 함께 반영하는 것이다.

---

## 2. 현재 확정된 모델 구조

현재 코드에서 실제로 사용되는 모델은 2개다.

1. LightGBM Classifier
2. LightGBM Ranker

과거에는 Random Forest를 넣은 3개 모델 실험이 있었지만,
현재 운영/포팅 코드 기준으로는 RF를 제거하고 2개 모델 구조를 사용한다.

| 모델 | 종류 | 입력 타깃 | 역할 | 비중 |
| --- | --- | --- | --- | ---: |
| LGBM Classifier | 분류 | `target` (상승 여부) | 절대 확률 생성 | 0.70 |
| LGBM Ranker | 랭킹 | `rank_target` (동일 날짜 종목 내 상대 순위) | 상대적 우위 생성 | 0.30 |

실제 구현은 다음과 같다.

```python
ordered["lgbm_probability"] = classifier.predict_proba(X)[:, 1]
rank_raw = ranker.predict(X)
ordered["return_rank"] = _cross_sectional_rank(rank_raw, ordered["date"])
ordered["ml_score"] = (
    CLASSIFIER_WEIGHT * ordered["lgbm_probability"]
    + RANKER_WEIGHT * ordered["return_rank"]
)
```

즉,

$$
ml\_score = 0.70 \cdot p_{clf} + 0.30 \cdot rank\_score
$$

이다.

---

## 3. 타깃 정의

### 3-1. 운영 코드의 기본 타깃
현재 운영 코드의 표준 타깃은 `future_excess_return_10D` 기반이다.

- 파일: [ml/features.py](ml/features.py)
- 핵심 로직:

```python
panel["future_return_10D"] = panel.groupby("ticker")["Close"].transform(
    lambda x: x.shift(-horizon) / x - 1
)

panel["future_excess_return_10D"] = panel["future_return_10D"] - benchmark
panel["target"] = (panel["future_excess_return_10D"] > EXCESS_RETURN_THRESHOLD).astype("Int64")
```

여기서 `EXCESS_RETURN_THRESHOLD` 는 [ml/config.py](ml/config.py) 에 정의되며,
현재 값은 0.03 (3%) 이다.

즉,

- 10영업일 뒤 초과수익률이 3% 이상이면 1
- 그렇지 않으면 0

으로 라벨링한다.

### 3-2. 실험 코드의 3개월 타깃
3개월 실험용 타깃도 별도로 준비되어 있다.

- 파일: [ml/test/test_ML2.py](ml/test/test_ML2.py)
- 실험 로직:

```python
panel["future_return_3M"] = (
    panel.groupby("ticker")["Close"]
    .transform(lambda s: s.shift(-63) / s - 1)
)

panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark
panel["target_3M"] = (panel["future_excess_return_3M"] > 0.0).astype(int)
```

3개월 실험은 "3개월 뒤 초과수익률이 양수인지"를 학습 타깃으로 보겠다는 의미다.

다만 현재 프러덕션 설계는 기본적으로 10영업일 + 3% threshold를 표준으로 사용한다.

---

## 4. 입력 데이터 구성

FEATURE는 [ml/features.py](ml/features.py) 에 정의된 `FEATURE_COLUMNS`를 사용한다.

실제 구성은 다음 그룹으로 나뉜다.

### Momentum
- `ret_5`, `ret_20`, `ret_60`
- `sector_relative_momentum`

### Technical
- `rsi`, `adx`, `atr_pct`, `macd_pct`, `bollinger_position`, `realized_vol20`

### Volume / Liquidity
- `volume_ratio`, `turnover`, `liquidity`

### Flow
- `foreign_5_pct`, `foreign_20_pct`
- `institution_5_pct`, `institution_20_pct`

### Fundamental
- `roe`, `pbr`, `pbr_sector_rank`, `per`, `per_sector_rank`, `ev_ebitda_rank`, `earnings_growth`

### Market / Macro
- `kospi_trend`, `vkospi`, `usdkrw`, `nasdaq`, `sp500`, `sox`, `gold`, `oil`

### Cross-sectional ranking features
- `momentum_rank`, `flow_rank`, `value_rank`

핵심은 “섹터/시차/상대순위 기반 바이어스”를 줄이면서, 시계열과 cross-sectional 정보 둘 다 학습에 넣는 것이다.

---

## 5. 학습 방식

### 학습 유니버스
현재 코드에서는 `feature_panel`에서 `training_universe` 조건을 사용해 학습 대상 관측치를 걸러낸다.

- [ml/train_model.py](ml/train_model.py)

```python
if "training_universe" in feature_panel:
    feature_panel = feature_panel[feature_panel["training_universe"].fillna(False)].copy()
```

이후 `labelled = feature_panel.dropna(subset=["target", "future_excess_return_10D"])` 로 학습 데이터만 남긴다.

### 검증 방식
시간순 검증이 사용된다.

- `walk_forward_splits()`
- `TEST_SESSIONS`, `WALK_FORWARD_STEP`

즉, 무작위 split 대신 시계열 walk-forward를 쓰며,
과거 데이터로 학습하고 이후 기간으로 검증하는 구조다.

---

## 6. 점수 산출 방식

### Classifier 확률
```python
ordered["lgbm_probability"] = classifier.predict_proba(X)[:, 1]
```

### Ranker 순위
```python
rank_raw = ranker.predict(X)
ordered["return_rank"] = _cross_sectional_rank(rank_raw, ordered["date"])
```

여기서 `return_rank`는 동시점 종목들 사이의 상대적 순위로, 0~1 범위의 percentile rank다.

### 최종 ML Score
```python
ordered["ml_score"] = (
    CLASSIFIER_WEIGHT * ordered["lgbm_probability"]
    + RANKER_WEIGHT * ordered["return_rank"]
)
```

- `CLASSIFIER_WEIGHT = 0.70`
- `RANKER_WEIGHT = 0.30`

이 조합은 단순히 절대 확률만 쓰는 대신,
“같은 날짜 내 상대적 우위”를 반영해 투자 우선순위를 더 안정적으로 만든다.

---

## 7. 추천 후보 선정

최종 스코어는 [ml/train_model.py](ml/train_model.py) 에서 다음처럼 정렬되고 시장별로 Top N을 뽑는다.

```python
latest_predictions = latest_scored[prediction_columns].sort_values(
    "ml_score", ascending=False
)

latest_predictions["ml_rank"] = (
    latest_predictions.groupby("market").cumcount() + 1
)

top_selection = latest_predictions.groupby("market", group_keys=False).head(
    TOP_SELECTION_SIZE_PER_MARKET
)
```

`TOP_SELECTION_SIZE_PER_MARKET` 는 기본적으로 10이다.

즉, 각 시장별 상위 10개가 최종 추천 후보가 된다.

---

## 8. 출력 스키마

최종 출력은 [ml/ml_filter.py](ml/ml_filter.py) 에서 다음 필드를 반환한다.

```python
{
    "ticker": ticker,
    "company_name": company_name,
    "up_probability": probability,
    "classification_probability": probability,
    "ml_score": score,
    "ml_rank": ml_rank,
    "lgbm_probability": float(prediction["lgbm_probability"]),
    "return_rank": float(prediction["return_rank"]),
    "ml_pass": ml_rank <= 10,
}
```

즉, 사용자/트레이딩 노드는 다음을 사용한다.

- `ml_score`: 정렬 기준
- `ml_rank`: 시장별 순위
- `up_probability`: 예측 확률
- `return_rank`: 상대적 우위

---

## 9. 최종 정리

현재 코드 기준의 실전 ML Filter는 다음과 같이 요약할 수 있다.

- 타깃: 10영업일 후 3% 초과수익 여부 (`target`)
- 모델: LGBM Classifier + LGBM Ranker
- 스코어: `0.7 * classifier_probability + 0.3 * cross_section_rank`
- 후보 선택: 시장별 상위 10개
- 운용 목적: Top 10 추천 + 매수 후보 필터링

3개월 실험 버전은 별도 benchmark 스크립트에서 검증 가능하지만,
현재 생산 코드의 확정 설계는 위 구조를 기준으로 운영된다.

---

## 10. 실험 vs 운영의 구분

- 운영(기본): 10영업일 horizon, threshold 3%
- 실험(benchmark): 3개월 horizon, positive target

즉,

- 운영은 “실전 투자 필터”에 맞춘 설계
- 실험은 “신호 해석/모델 비교용 설계”

로 구분된다.



