# 3개월 Horizon ML Filter 전체 정리

## 1. 목표

이 설정은 10영업일 단기 타깃 대신, 63영업일(약 3개월) 뒤의 시장대비 초과수익률을 예측하는 구조를 사용한다.

핵심 목표는 다음과 같다.

- 같은 날짜의 종목들 간 상대적 우위(`rank_score`)를 반영한다.
- 절대 확률(`classifier_probability`)과 상대 랭킹(`rank_score`)를 합산하여 최종 점수(`ml_score`)를 만든다.
- 최종적으로 시장별 상위 15개를 추천 후보로 선택한다.
- ML Filter는 항상 ON 상태에서 동작한다.

---

## 2. 최종 점수 공식

현재 두 모델 앙상블의 가중치는 다음과 같다.

- classifier = 0.70
- ranker = 0.30

최종 점수는 아래와 같이 계산한다.

$$
ml\_score = 0.70 \cdot p_{clf} + 0.30 \cdot rank\_score
$$

- `p_clf`: LightGBM classifier의 절대 상승 확률
- `rank_score`: 동시점 종목들 사이의 상대적 순위 점수

---

## 3. 타깃 정의

### 3개월 시장대비 초과수익률 목표

```python
panel["future_return_3M"] = (
    panel.groupby("ticker")["Close"].transform(
        lambda s: s.shift(-63) / s - 1
    )
)
```

시장 대비 초과수익률은 다음으로 계산한다.

```python
panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark
panel["target_3M"] = (panel["future_excess_return_3M"] > 0.0).astype("Int64")
```

즉,

- 63영업일 뒤 시장 대비 초과수익률이 양수면 1
- 그렇지 않으면 0

으로 레이블링한다.

---

## 4. 학습 구조

### 4.1 Fold 설정

요청한 구조는 다음과 같다.

- 학습 기간: 24개월
- validation 기간: 3개월
- walk-forward step: 3개월
- horizon: 63영업일

즉,

```python
train_months = 24
valid_months = 3
step_months = 3
horizon = 63
```

으로 동작한다.

### 4.2 Walk-forward 방식

모델은 랜덤 split이 아니라 시간 순서 기반 walk-forward validation을 사용한다.

- 과거 24개월로 학습
- 다음 3개월을 validation
- 3개월씩 이동하면서 재검증

이 방식은 시계열 데이터에서 leakage를 줄이고 실무 투자 흐름에 더 적합하다.

---

## 5. 모델 구성

### Classifier

- LightGBM Classifier
- 입력: `FEATURE_COLUMNS`
- 타깃: `target_3M`
- 역할: 절대 상승 확률 생성

### Ranker

- LightGBM Ranker
- 입력: 동일 feature set
- 타깃: `rank_target_3M`
- 역할: 같은 날짜 내 종목간 상대 순위 학습

### 최종 앙상블

```python
valid_df["ml_score"] = (
    CLASSIFIER_WEIGHT * valid_df["probability"]
    + RANKER_WEIGHT * valid_df["rank_score"]
)
```

---

## 6. 추천 후보 흐름

추천 엔진은 아래 순서로 동작한다.

1. 후보군 생성
   - ML snapshot 기반 후보 선택
   - ML Filter는 항상 ON
2. 종목별 ML 점수 계산
   - classifier probability
   - cross-sectional rank_score
   - 최종 ml_score
3. 후보군 정렬
   - 상위 등급 순으로 정렬
4. Top15 선별
   - 시장별/전체 기준으로 상위 15개만 저장
5. 최종 추천 저장
   - `save_recommendations()` 로 보관

---

## 7. 요약 리포트 함수

실제 재학습 후 요약용 데이터를 보려면 아래 함수를 사용한다.

```python
from ml.train_model import summarize_training_report, train_market_ensemble

artifact = train_market_ensemble(save=True, progress=True)
summary = summarize_training_report(artifact)
print(summary)
```

또는 실행 파일로 바로 재학습 가능하다.

```bash
python scripts/retrain_3m_ensemble.py
```

생성되는 요약 파일은 다음 경로다.

- `data/ml/reports/3m_ensemble_summary.json`

---

## 8. 실행 스크립트

실제 재학습용 실행 파일은 다음과 같다.

- [scripts/retrain_3m_ensemble.py](scripts/retrain_3m_ensemble.py)

이 스크립트는 다음을 수행한다.

- 모델 재학습
- 결과 저장
- 요약 JSON 생성
- 핵심 metric 출력

---

## 9. 정리

현재 구조는 단순히 “상승 확률 예측”에만 머무르지 않고,
다음 두 가지를 함께 반영하는 실전형 투자 필터다.

- 절대 분류 신호: probability
- 상대 순위 신호: rank_score

이 조합을 통해 투자 후보를 더 안정적으로 Top15로 뽑을 수 있다.

현재 운영 설계의 핵심은 다음과 같이 요약할 수 있다.

- 타깃: 63영업일 뒤 시장대비 초과수익률 > 0
- 학습: 24개월 train / 3개월 validation / 3개월 step walk-forward
- 앙상블: 0.70 classifier + 0.30 ranker
- 추천수: Top 15
