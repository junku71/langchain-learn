import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from ml.config import FEATURE_PANEL_PATH
from ml.features import FEATURE_COLUMNS
from ml.storage import read_frame

# -----------------------------
# 1) 1개월 future target 만들기
# -----------------------------
def add_3m_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()

    # 3개월 보유기간 = 약 63 거래일
    horizon = 63

    # 종목별 3개월 후 수익률
    panel["future_return_3M"] = (
        panel.groupby("ticker")["Close"]
        .transform(lambda s: s.shift(-horizon) / s - 1)
    )

    # 시장 벤치마크
    if "kospi_future_return" in panel.columns and "kosdaq_future_return" in panel.columns:
        market_name = panel.get("market", pd.Series("KOSPI", index=panel.index))
        benchmark = panel["kospi_future_return"].where(
            market_name.ne("KOSDAQ"), panel["kosdaq_future_return"]
        )
    else:
        benchmark = 0.0

    panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark
    panel["target_3M"] = (panel["future_excess_return_3M"] > 0).astype(int)

    return panel

# -----------------------------
# 2) rolling train/valid split
# -----------------------------
def make_rolling_folds(
    panel: pd.DataFrame,
    train_months: int = 24,
    valid_months: int = 3,
    step_months: int = 3,
):
    """
    train_months = 24 (약 2년)
    valid_months = 3 (약 3개월)
    step_months = 3 (매 3개월마다 fold 이동)
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    train_days = train_months * 21
    valid_days = valid_months * 21
    step_days = step_months * 21

    folds = []
    for start in range(0, len(dates) - train_days - valid_days + 1, step_days):
        train_slice = dates[start:start + train_days]
        valid_slice = dates[start + train_days:start + train_days + valid_days]

        train = panel[panel["date"].isin(train_slice)].copy()
        valid = panel[panel["date"].isin(valid_slice)].copy()

        if train.empty or valid.empty:
            continue

        folds.append((train, valid))

    return folds

# -----------------------------
# 3) 모델 훈련/평가
# -----------------------------
def train_lgbm_folds(panel: pd.DataFrame):
    panel = add_3m_targets(panel)

    # 사용할 feature 컬럼
    feature_cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    if not feature_cols:
        raise ValueError("Feature columns not found in panel")

    folds = make_rolling_folds(panel)

    fold_results = []
    for idx, (train, valid) in enumerate(folds, 1):
        X_train = train[feature_cols].copy().fillna(train[feature_cols].median())
        y_train = train["target_3M"].astype(int)

        X_valid = valid[feature_cols].copy().fillna(train[feature_cols].median())
        y_valid = valid["target_3M"].astype(int)

        model = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_valid)[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y_valid, prob)
        pr_auc = average_precision_score(y_valid, prob)
        acc = accuracy_score(y_valid, pred)

        fold_results.append({
            "fold": idx,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "valid_start": valid["date"].min(),
            "valid_end": valid["date"].max(),
            "AUC": auc,
            "PR_AUC": pr_auc,
            "ACC": acc,
        })

    return fold_results

# -----------------------------
# 4) 실행 예시
# -----------------------------
if __name__ == "__main__":
    panel = read_frame(FEATURE_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    results = train_lgbm_folds(panel)

    for row in results:
        print(row)

    avg_auc = np.mean([r["AUC"] for r in results])
    avg_pr_auc = np.mean([r["PR_AUC"] for r in results])
    print(f"\nAverage AUC: {avg_auc:.4f}")
    print(f"Average PR-AUC: {avg_pr_auc:.4f}")