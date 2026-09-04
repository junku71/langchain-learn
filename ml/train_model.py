import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score, brier_score_loss, ndcg_score
from sklearn.pipeline import Pipeline

from analysis.ticker_mapper import get_company_name, get_yfinance_ticker
from ml.features import FEATURE_COLUMNS, create_panel_features
from ml.build_dataset import build_dataset
from ml.config import (
    FEATURE_IMPORTANCE_PATH,
    FEATURE_PANEL_PATH,
    LGBM_CLASSIFIER_WEIGHT,
    LGBM_RANKER_WEIGHT,
    MODEL_REPORT_PATH,
    OOS_PREDICTIONS_PATH,
    PREDICTION_HORIZON,
    RF_WEIGHT as CONFIG_RF_WEIGHT,
    TOP_SELECTION_SIZE_PER_MARKET,
    TOP_SELECTION_PATH,
)
from ml.storage import read_frame, write_frame
from ml.validate_dataset import validate_dataset


MODEL_DIR = Path("ml/models")
MODEL_PATH = MODEL_DIR / "korea_top200_ensemble.joblib"
MODEL_VERSION = 8
TEST_SESSIONS = 126
WALK_FORWARD_STEP = 21
CLASSIFIER_WEIGHT = LGBM_CLASSIFIER_WEIGHT
RANKER_WEIGHT = LGBM_RANKER_WEIGHT
RF_WEIGHT = CONFIG_RF_WEIGHT


def resolve_stock(stock_name_or_ticker: str) -> tuple[str, str]:
    value = stock_name_or_ticker.strip()
    upper = value.upper()
    if re.fullmatch(r"\d{6}", upper):
        ticker = f"{upper}.KS"
    elif re.fullmatch(r"\d{6}\.(KS|KQ)", upper):
        ticker = upper
    else:
        ticker = get_yfinance_ticker(value)
    return ticker, get_company_name(ticker)


def create_classifier() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def create_ranker() -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def create_random_forest() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _rank_groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _cross_sectional_rank(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    series = pd.Series(values, index=dates.index)
    return series.groupby(dates).rank(pct=True).fillna(0.5).to_numpy()


def walk_forward_splits(panel: pd.DataFrame):
    dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    if len(dates) <= TEST_SESSIONS + PREDICTION_HORIZON + 60:
        raise ValueError("Panel needs more history for purged 4.5y/0.5y evaluation")
    first_test_index = len(dates) - TEST_SESSIONS
    for fold_start in range(first_test_index, len(dates), WALK_FORWARD_STEP):
        fold_end = min(fold_start + WALK_FORWARD_STEP, len(dates))
        train_end_index = fold_start - PREDICTION_HORIZON
        train = panel[panel["date"] < dates[train_end_index]].copy()
        test_dates = dates[fold_start:fold_end]
        test = panel[panel["date"].isin(test_dates)].copy()
        if not train.empty and not test.empty:
            yield train, test


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _fit_models(
    train: pd.DataFrame,
    *,
    progress_label: str | None = None,
) -> tuple[Any, Any, Any]:
    ordered = train.sort_values(["date", "ticker"])
    X = ordered[FEATURE_COLUMNS]
    y = ordered["target"].astype(int)
    label = f"{progress_label} " if progress_label else ""

    step_started = time.monotonic()
    if progress_label:
        print(f"{label}LightGBM classifier training...", flush=True)
    classifier = create_classifier().fit(X, y)
    if progress_label:
        print(
            f"{label}LightGBM classifier done "
            f"({_format_duration(time.monotonic() - step_started)})",
            flush=True,
        )

    step_started = time.monotonic()
    if progress_label:
        print(f"{label}LightGBM ranker training...", flush=True)
    ranker = create_ranker().fit(
        X,
        ordered["rank_target"].astype(int),
        group=_rank_groups(ordered),
    )
    if progress_label:
        print(
            f"{label}LightGBM ranker done "
            f"({_format_duration(time.monotonic() - step_started)})",
            flush=True,
        )

    step_started = time.monotonic()
    if progress_label:
        print(f"{label}Random Forest training...", flush=True)
    random_forest = create_random_forest().fit(X, y)
    if progress_label:
        print(
            f"{label}Random Forest done "
            f"({_format_duration(time.monotonic() - step_started)})",
            flush=True,
        )
    return classifier, ranker, random_forest


def _predict_components(
    models: tuple[Any, Any, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    classifier, ranker, random_forest = models
    ordered = frame.sort_values(["date", "ticker"]).copy()
    X = ordered[FEATURE_COLUMNS]
    ordered["lgbm_probability"] = classifier.predict_proba(X)[:, 1]
    ordered["rf_probability"] = random_forest.predict_proba(X)[:, 1]
    classification_weight = CLASSIFIER_WEIGHT + RF_WEIGHT
    ordered["classification_probability"] = (
        CLASSIFIER_WEIGHT * ordered["lgbm_probability"]
        + RF_WEIGHT * ordered["rf_probability"]
    ) / classification_weight
    rank_raw = ranker.predict(X)
    ordered["return_rank"] = _cross_sectional_rank(rank_raw, ordered["date"])
    ordered["ml_score"] = (
        CLASSIFIER_WEIGHT * ordered["lgbm_probability"]
        + RANKER_WEIGHT * ordered["return_rank"]
        + RF_WEIGHT * ordered["rf_probability"]
    )
    return ordered


def _safe_auc(target: pd.Series, prediction: pd.Series) -> float | None:
    return float(roc_auc_score(target, prediction)) if target.nunique() == 2 else None


def _mean_daily_ndcg(frame: pd.DataFrame, k: int = 10) -> float | None:
    scores = []
    for _, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        relevance = group["future_excess_return_10D"].rank(method="average").to_numpy()
        prediction = group["ml_score"].to_numpy()
        scores.append(float(ndcg_score([relevance], [prediction], k=min(k, len(group)))))
    return float(np.mean(scores)) if scores else None


def _universe_hash(universe: pd.DataFrame) -> str:
    tickers = "|".join(sorted(universe["ticker"].astype(str)))
    return hashlib.sha256(tickers.encode("utf-8")).hexdigest()


def dataset_fingerprint(panel: pd.DataFrame) -> str:
    columns = [column for column in ("date", "ticker", "market_cap_rank") if column in panel]
    ordered = panel[columns].sort_values(columns).reset_index(drop=True)
    digest = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha256(digest).hexdigest()


def _top_selection_metrics(frame: pd.DataFrame, top: int = 10) -> dict[str, float | None]:
    group_columns = ["date", "market"] if "market" in frame else ["date"]
    selected = frame.sort_values(
        [*group_columns, "ml_score"], ascending=[True] * len(group_columns) + [False]
    ).groupby(group_columns, sort=False).head(top)
    if selected.empty:
        return {"top10_mean_excess_return": None, "top10_hit_rate": None}
    returns = selected["future_excess_return_10D"]
    return {
        "top10_mean_excess_return": float(returns.mean()),
        "top10_hit_rate": float(returns.gt(0).mean()),
    }


def _feature_importance(models: tuple[Any, Any, Any]) -> pd.DataFrame:
    classifier, ranker, random_forest = models
    values = {
        "feature": FEATURE_COLUMNS,
        "lgbm_classifier": classifier.feature_importances_,
        "lgbm_ranker": ranker.feature_importances_,
        "random_forest": random_forest.named_steps["model"].feature_importances_,
    }
    result = pd.DataFrame(values)
    for column in ("lgbm_classifier", "lgbm_ranker", "random_forest"):
        total = result[column].sum()
        result[column] = result[column] / total if total else 0.0
    result["weighted_importance"] = (
        CLASSIFIER_WEIGHT * result["lgbm_classifier"]
        + RANKER_WEIGHT * result["lgbm_ranker"]
        + RF_WEIGHT * result["random_forest"]
    )
    return result.sort_values("weighted_importance", ascending=False)


def train_market_ensemble(
    price_panel: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    economy_df: pd.DataFrame | None = None,
    *,
    save: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    training_started = time.monotonic()
    if price_panel is None or universe is None or economy_df is None:
        feature_panel = read_frame(FEATURE_PANEL_PATH)
        if feature_panel.empty:
            feature_panel = build_dataset()
        validate_dataset(feature_panel, require_training_ready=True)
        universe = feature_panel[[
            column for column in ("ticker", "market", "sector") if column in feature_panel
        ]].drop_duplicates("ticker")
    else:
        feature_panel = create_panel_features(price_panel, economy_df, universe)
    if "training_universe" in feature_panel:
        feature_panel = feature_panel[feature_panel["training_universe"].fillna(False)].copy()
    labelled = feature_panel.dropna(subset=["target", "future_excess_return_10D"])
    evaluations = []
    first_train_rows = 0
    total_folds = (TEST_SESSIONS + WALK_FORWARD_STEP - 1) // WALK_FORWARD_STEP
    total_training_steps = total_folds + 1  # Walk-forward folds plus final fit.
    if progress:
        first_date = pd.Timestamp(labelled["date"].min()).date()
        last_date = pd.Timestamp(labelled["date"].max()).date()
        print(
            f"Training started: {len(labelled):,} labelled rows, "
            f"{labelled['ticker'].nunique():,} tickers, {first_date}..{last_date}",
            flush=True,
        )
        print(
            f"Plan: {total_folds} walk-forward folds + 1 final model fit",
            flush=True,
        )
    for fold_number, (train, test) in enumerate(walk_forward_splits(labelled), 1):
        fold_started = time.monotonic()
        if fold_number == 1:
            first_train_rows = len(train)
        if progress:
            test_start = pd.Timestamp(test["date"].min()).date()
            test_end = pd.Timestamp(test["date"].max()).date()
            print(
                f"[{fold_number}/{total_training_steps}] Walk-forward fold "
                f"{fold_number}/{total_folds}: train={len(train):,}, "
                f"test={len(test):,} ({test_start}..{test_end})",
                flush=True,
            )
        fold_result = _predict_components(
            _fit_models(
                train,
                progress_label=(
                    f"[{fold_number}/{total_training_steps}]" if progress else None
                ),
            ),
            test,
        )
        fold_result["walk_forward_fold"] = fold_number
        evaluations.append(fold_result)
        if progress:
            elapsed = time.monotonic() - training_started
            average_step = elapsed / fold_number
            remaining_steps = total_training_steps - fold_number
            eta = average_step * remaining_steps
            print(
                f"[{fold_number}/{total_training_steps}] Fold complete: "
                f"fold={_format_duration(time.monotonic() - fold_started)}, "
                f"elapsed={_format_duration(elapsed)}, "
                f"estimated remaining={_format_duration(eta)}",
                flush=True,
            )
    if not evaluations:
        raise ValueError("No valid walk-forward folds were created")
    evaluation = pd.concat(evaluations, ignore_index=True)
    selection_metrics = _top_selection_metrics(evaluation)
    metrics = {
        "lgbm_classifier_auc": _safe_auc(
            evaluation["target"], evaluation["lgbm_probability"]
        ),
        "random_forest_auc": _safe_auc(
            evaluation["target"], evaluation["rf_probability"]
        ),
        "ensemble_auc": _safe_auc(evaluation["target"], evaluation["ml_score"]),
        "ensemble_pr_auc": float(average_precision_score(evaluation["target"], evaluation["ml_score"])),
        "classification_brier": float(brier_score_loss(
            evaluation["target"], evaluation["classification_probability"]
        )),
        "ensemble_accuracy": float(
            accuracy_score(evaluation["target"], evaluation["classification_probability"] >= 0.5)
        ),
        "rank_spearman": float(
            evaluation["return_rank"].corr(
                evaluation["future_excess_return_10D"], method="spearman"
            )
        ),
        "ndcg_at_10": _mean_daily_ndcg(evaluation, 10),
        "initial_train_rows": first_train_rows,
        "test_rows": len(evaluation),
        "test_start": str(evaluation["date"].min().date()),
        "walk_forward_folds": len(evaluations),
        "walk_forward_step": WALK_FORWARD_STEP,
        **selection_metrics,
    }

    final_step = total_folds + 1
    if progress:
        print(
            f"[{final_step}/{total_training_steps}] Training final models on "
            f"all {len(labelled):,} labelled rows...",
            flush=True,
        )
    final_models = _fit_models(
        labelled,
        progress_label=(
            f"[{final_step}/{total_training_steps}]" if progress else None
        ),
    )
    latest_date = feature_panel["date"].max()
    latest = feature_panel[feature_panel["date"].eq(latest_date)]
    if "prediction_universe" in latest:
        latest = latest[latest["prediction_universe"].fillna(False)]
    latest_scored = _predict_components(final_models, latest)
    prediction_columns = [column for column in (
        "ticker", "name", "market", "sector", "lgbm_probability", "return_rank", "rf_probability",
        "classification_probability", "ml_score",
    ) if column in latest_scored]
    latest_predictions = latest_scored[prediction_columns].sort_values(
        "ml_score", ascending=False
    )
    latest_predictions["global_ml_rank"] = range(1, len(latest_predictions) + 1)
    latest_predictions["ml_rank"] = (
        latest_predictions.groupby("market").cumcount() + 1
        if "market" in latest_predictions
        else latest_predictions["global_ml_rank"]
    )
    top_selection = (
        latest_predictions.groupby("market", group_keys=False)
        .head(TOP_SELECTION_SIZE_PER_MARKET)
        if "market" in latest_predictions
        else latest_predictions.head(TOP_SELECTION_SIZE_PER_MARKET)
    )
    artifact = {
        "models": {
            "lgbm_classifier": final_models[0],
            "lgbm_ranker": final_models[1],
            "random_forest": final_models[2],
        },
        "latest_predictions": latest_predictions.set_index("ticker").to_dict("index"),
        "top_selection": top_selection.to_dict("records"),
        "metadata": {
            "model_version": MODEL_VERSION,
            "algorithm": "LGBMClassifier+LGBMRanker+RandomForestClassifier",
            "trained_through": str(pd.Timestamp(latest_date).date()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "prediction_horizon": PREDICTION_HORIZON,
            "feature_columns": FEATURE_COLUMNS,
            "universe_size": int(universe["ticker"].nunique()),
            "universe_hash": _universe_hash(universe),
            "dataset_fingerprint": dataset_fingerprint(feature_panel),
            "weights": {
                "lgbm_classifier": CLASSIFIER_WEIGHT,
                "lgbm_ranker": RANKER_WEIGHT,
                "random_forest": RF_WEIGHT,
            },
            "metrics": metrics,
        },
    }
    if save:
        if progress:
            print("Saving model and evaluation artifacts...", flush=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        MODEL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_REPORT_PATH.write_text(
            json.dumps(artifact["metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _feature_importance(final_models).to_csv(
            FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig"
        )
        write_frame(evaluation[[column for column in (
            "date", "ticker", "target", "future_excess_return_10D", "lgbm_probability",
            "rf_probability", "classification_probability", "return_rank", "ml_score",
            "walk_forward_fold",
        ) if column in evaluation]], OOS_PREDICTIONS_PATH)
        top_selection.to_csv(TOP_SELECTION_PATH, index=False, encoding="utf-8-sig")
    if progress:
        print(
            f"Training complete in {_format_duration(time.monotonic() - training_started)}",
            flush=True,
        )
    artifact["model_path"] = str(MODEL_PATH)
    return artifact


def train_and_save_model() -> dict[str, Any]:
    artifact = train_market_ensemble()
    print(artifact["metadata"])
    print("Saved:", artifact["model_path"])
    return artifact


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    train_and_save_model()
