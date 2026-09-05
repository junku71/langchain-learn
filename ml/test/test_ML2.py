

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from ml.config import FEATURE_PANEL_PATH
from ml.features import FEATURE_COLUMNS
from ml.storage import read_frame

# Two-model ensemble weights
# classifier = 0.70, ranker = 0.30
# - classifier: absolute probability signal
# - ranker: cross-sectional relative ranking signal
CLASSIFIER_WEIGHT = 0.70
RANKER_WEIGHT = 0.30


def add_3m_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    horizon = 63

    panel["future_return_3M"] = (
        panel.groupby("ticker")["Close"]
        .transform(lambda s: s.shift(-horizon) / s - 1)
    )

    if "kospi_future_return" in panel.columns and "kosdaq_future_return" in panel.columns:
        market_name = panel.get("market", pd.Series("KOSPI", index=panel.index))
        benchmark = panel["kospi_future_return"].where(
            market_name.ne("KOSDAQ"), panel["kosdaq_future_return"]
        )
    else:
        benchmark = 0.0

    panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark
    panel["target_3M"] = (panel["future_excess_return_3M"] > 0.0).astype(int)
    return panel


def make_rolling_folds(panel: pd.DataFrame, train_months=24, valid_months=3, step_months=3):
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


def _cross_sectional_rank(values: pd.Series, date_series: pd.Series) -> pd.Series:
    df = pd.DataFrame({"date": date_series, "value": values})
    return df.groupby("date")["value"].rank(method="average", pct=True).fillna(0.5)


def train_three_model_folds(panel: pd.DataFrame):
    panel = add_3m_targets(panel)
    feature_cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    if not feature_cols:
        raise ValueError("Feature columns not found")

    fold_results = []
    for idx, (train, valid) in enumerate(make_rolling_folds(panel), 1):
        X_train = train[feature_cols].copy().fillna(train[feature_cols].median())
        X_valid = valid[feature_cols].copy().fillna(train[feature_cols].median())
        y_train = train["target_3M"].astype(int)
        y_valid = valid["target_3M"].astype(int)

        classifier = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
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
        classifier.fit(X_train, y_train)

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
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

        group_sizes = train.groupby("date").size().tolist()
        ranker.fit(X_train, y_train, group=group_sizes)

        valid_df = valid.copy()
        valid_df["lgbm_probability"] = classifier.predict_proba(X_valid)[:, 1]

        rank_raw = ranker.predict(X_valid)
        valid_df["rank_score"] = _cross_sectional_rank(pd.Series(rank_raw, index=valid_df.index), valid_df["date"])

        # Final composite score: classifier + relative rank
        valid_df["ml_score"] = (
            CLASSIFIER_WEIGHT * valid_df["lgbm_probability"]
            + RANKER_WEIGHT * valid_df["rank_score"]
        )

        pred = (valid_df["ml_score"] >= 0.65).astype(int)

        auc = roc_auc_score(y_valid, valid_df["ml_score"])
        pr_auc = average_precision_score(y_valid, valid_df["ml_score"])
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
            "weights": {
                "classifier": CLASSIFIER_WEIGHT,
                "ranker": RANKER_WEIGHT,
            },
        })

    return fold_results


if __name__ == "__main__":
    panel = read_frame(FEATURE_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    results = train_three_model_folds(panel)

    for row in results:
        print(row)

    avg_auc = np.mean([r["AUC"] for r in results])
    avg_pr_auc = np.mean([r["PR_AUC"] for r in results])
    avg_acc = np.mean([r["ACC"] for r in results])
    print(f"\nAverage AUC: {avg_auc:.4f}")
    print(f"Average PR-AUC: {avg_pr_auc:.4f}")
    print(f"Average ACC: {avg_acc:.4f}")
    print(f"Weights: classifier={CLASSIFIER_WEIGHT}, ranker={RANKER_WEIGHT}")