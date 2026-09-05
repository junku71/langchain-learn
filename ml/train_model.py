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
THREE_MONTH_HORIZON = 63
TRAIN_MONTHS = 24
VALIDATION_MONTHS = 3
WALK_FORWARD_STEP_MONTHS = 3
CLASSIFIER_WEIGHT = LGBM_CLASSIFIER_WEIGHT
RANKER_WEIGHT = LGBM_RANKER_WEIGHT


def add_3m_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    horizon = THREE_MONTH_HORIZON

    panel["future_return_3M"] = (
        panel.groupby("ticker")["Close"].transform(
            lambda s: s.shift(-horizon) / s - 1
        )
    )

    if "kospi_future_return" in panel.columns and "kosdaq_future_return" in panel.columns:
        market_name = panel.get("market", pd.Series("KOSPI", index=panel.index))
        benchmark = panel["kospi_future_return"].where(
            market_name.ne("KOSDAQ"), panel["kosdaq_future_return"]
        )
    else:
        benchmark = 0.0

    panel["future_excess_return_3M"] = panel["future_return_3M"] - benchmark
    panel["target_3M"] = (panel["future_excess_return_3M"] > 0.0).astype("Int64")
    panel["rank_target_3M"] = (
        panel.groupby("date")["future_excess_return_3M"]
        .rank(method="average", pct=True)
        .mul(9)
        .round()
        .astype("Int64")
    )
    return panel


def make_rolling_folds(
    panel: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    valid_months: int = VALIDATION_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
):
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


def train_lgbm_rank_folds(panel: pd.DataFrame):
    panel = add_3m_targets(panel)
    feature_cols = [column for column in FEATURE_COLUMNS if column in panel.columns]
    if not feature_cols:
        raise ValueError("Feature columns not found")

    fold_results = []
    for index, (train, valid) in enumerate(make_rolling_folds(panel), 1):
        X_train = train[feature_cols].copy().fillna(train[feature_cols].median())
        y_train = train["target_3M"].astype(int)

        X_valid = valid[feature_cols].copy().fillna(train[feature_cols].median())
        y_valid = valid["target_3M"].astype(int)

        model = LGBMClassifier(
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

        model.fit(X_train, y_train)

        probability = model.predict_proba(X_valid)[:, 1]

        valid_df = valid.copy()
        valid_df["probability"] = probability
        valid_df["rank_score"] = (
            valid_df.groupby("date")["probability"]
            .rank(method="average", pct=True)
            .fillna(0.5)
        )
        valid_df["ml_score"] = (
            CLASSIFIER_WEIGHT * valid_df["probability"]
            + RANKER_WEIGHT * valid_df["rank_score"]
        )
        pred = (valid_df["ml_score"] >= 0.5).astype(int)

        auc = roc_auc_score(y_valid, valid_df["ml_score"])
        pr_auc = average_precision_score(y_valid, valid_df["ml_score"])
        acc = accuracy_score(y_valid, pred)

        fold_results.append({
            "fold": index,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "valid_start": valid["date"].min(),
            "valid_end": valid["date"].max(),
            "AUC": auc,
            "PR_AUC": pr_auc,
            "ACC": acc,
        })

    return fold_results


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
    train_days = TRAIN_MONTHS * 21
    valid_days = VALIDATION_MONTHS * 21
    step_days = WALK_FORWARD_STEP_MONTHS * 21

    if len(dates) < train_days + valid_days:
        raise ValueError(
            "Panel is too short for a 24-month train + 3-month validation walk-forward setup"
        )

    for start in range(0, len(dates) - train_days - valid_days + 1, step_days):
        train_slice = dates[start:start + train_days]
        valid_slice = dates[start + train_days:start + train_days + valid_days]

        train = panel[panel["date"].isin(train_slice)].copy()
        test = panel[panel["date"].isin(valid_slice)].copy()

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
) -> tuple[Any, Any]:
    ordered = train.sort_values(["date", "ticker"]).copy()
    X = ordered[FEATURE_COLUMNS]
    target_column = "target_3M" if "target_3M" in ordered else "target"
    y = ordered[target_column].astype(int)
    rank_target_column = "rank_target_3M" if "rank_target_3M" in ordered else "rank_target"
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
        ordered[rank_target_column].astype(int),
        group=_rank_groups(ordered),
    )
    if progress_label:
        print(
            f"{label}LightGBM ranker done "
            f"({_format_duration(time.monotonic() - step_started)})",
            flush=True,
        )

    return classifier, ranker


def _predict_components(
    models: tuple[Any, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    classifier, ranker = models
    ordered = frame.sort_values(["date", "ticker"]).copy()
    X = ordered[FEATURE_COLUMNS]
    ordered["lgbm_probability"] = classifier.predict_proba(X)[:, 1]
    ordered["classification_probability"] = ordered["lgbm_probability"]
    rank_raw = ranker.predict(X)
    ordered["return_rank"] = _cross_sectional_rank(rank_raw, ordered["date"])
    ordered["ml_score"] = (
        CLASSIFIER_WEIGHT * ordered["lgbm_probability"]
        + RANKER_WEIGHT * ordered["return_rank"]
    )
    return ordered


def _safe_auc(target: pd.Series, prediction: pd.Series) -> float | None:
    return float(roc_auc_score(target, prediction)) if target.nunique() == 2 else None


def _mean_daily_ndcg(frame: pd.DataFrame, k: int = 10) -> float | None:
    target_column = "future_excess_return_3M" if "future_excess_return_3M" in frame else "future_excess_return_10D"
    scores = []
    for _, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        relevance = group[target_column].rank(method="average").to_numpy()
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


def _top_selection_metrics(frame: pd.DataFrame, top: int = 15) -> dict[str, float | None]:
    target_column = "future_excess_return_3M" if "future_excess_return_3M" in frame else "future_excess_return_10D"
    group_columns = ["date", "market"] if "market" in frame else ["date"]
    selected = frame.sort_values(
        [*group_columns, "ml_score"], ascending=[True] * len(group_columns) + [False]
    ).groupby(group_columns, sort=False).head(top)
    if selected.empty:
        return {"top15_mean_excess_return": None, "top15_hit_rate": None}
    returns = selected[target_column]
    return {
        "top15_mean_excess_return": float(returns.mean()),
        "top15_hit_rate": float(returns.gt(0).mean()),
    }


def _feature_importance(models: tuple[Any, Any]) -> pd.DataFrame:
    classifier, ranker = models
    values = {
        "feature": FEATURE_COLUMNS,
        "lgbm_classifier": classifier.feature_importances_,
        "lgbm_ranker": ranker.feature_importances_,
    }
    result = pd.DataFrame(values)
    for column in ("lgbm_classifier", "lgbm_ranker"):
        total = result[column].sum()
        result[column] = result[column] / total if total else 0.0
    result["weighted_importance"] = (
        CLASSIFIER_WEIGHT * result["lgbm_classifier"]
        + RANKER_WEIGHT * result["lgbm_ranker"]
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
    feature_panel = add_3m_targets(feature_panel)
    labelled = feature_panel.dropna(subset=["target_3M", "future_excess_return_3M"])
    evaluations = []
    first_train_rows = 0
    folds = list(walk_forward_splits(labelled))
    total_folds = len(folds)
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
    for fold_number, (train, test) in enumerate(folds, 1):
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
            evaluation["target_3M"], evaluation["lgbm_probability"]
        ),
        "ensemble_auc": _safe_auc(evaluation["target_3M"], evaluation["ml_score"]),
        "ensemble_pr_auc": float(average_precision_score(evaluation["target_3M"], evaluation["ml_score"])),
        "classification_brier": float(brier_score_loss(
            evaluation["target_3M"], evaluation["classification_probability"]
        )),
        "ensemble_accuracy": float(
            accuracy_score(evaluation["target_3M"], evaluation["classification_probability"] >= 0.5)
        ),
        "rank_spearman": float(
            evaluation["return_rank"].corr(
                evaluation["future_excess_return_3M"], method="spearman"
            )
        ),
        "ndcg_at_10": _mean_daily_ndcg(evaluation, 10),
        "initial_train_rows": first_train_rows,
        "test_rows": len(evaluation),
        "test_start": str(evaluation["date"].min().date()),
        "walk_forward_folds": len(evaluations),
        "walk_forward_step": WALK_FORWARD_STEP_MONTHS * 21,
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
        "ticker", "name", "market", "sector",
        "lgbm_probability", "return_rank",
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
        },
        "latest_predictions": latest_predictions.set_index("ticker").to_dict("index"),
        "top_selection": top_selection.to_dict("records"),
        "metadata": {
            "model_version": MODEL_VERSION,
            "algorithm": "LGBMClassifier+LGBMRanker",
            "trained_through": str(pd.Timestamp(latest_date).date()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "prediction_horizon": THREE_MONTH_HORIZON,
            "feature_columns": FEATURE_COLUMNS,
            "universe_size": int(universe["ticker"].nunique()),
            "universe_hash": _universe_hash(universe),
            "dataset_fingerprint": dataset_fingerprint(feature_panel),
            "weights": {
                "lgbm_classifier": CLASSIFIER_WEIGHT,
                "lgbm_ranker": RANKER_WEIGHT,
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
            "date", "ticker", "target_3M", "future_excess_return_3M",
            "lgbm_probability", "classification_probability", "return_rank", "ml_score",
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


def summarize_training_report(
    artifact: dict[str, Any] | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if artifact is None:
        artifact = train_market_ensemble(save=False, progress=False)

    metadata = artifact.get("metadata", {})
    metrics = metadata.get("metrics", {})
    weights = metadata.get("weights", {
        "lgbm_classifier": CLASSIFIER_WEIGHT,
        "lgbm_ranker": RANKER_WEIGHT,
    })

    summary = {
        "model_version": metadata.get("model_version"),
        "trained_through": metadata.get("trained_through"),
        "prediction_horizon_days": int(metadata.get("prediction_horizon", THREE_MONTH_HORIZON)),
        "train_months": TRAIN_MONTHS,
        "validation_months": VALIDATION_MONTHS,
        "walk_forward_step_months": WALK_FORWARD_STEP_MONTHS,
        "ensemble_weights": {
            "classifier": float(weights.get("lgbm_classifier", CLASSIFIER_WEIGHT)),
            "ranker": float(weights.get("lgbm_ranker", RANKER_WEIGHT)),
        },
        "metrics": {
            "AUC": metrics.get("ensemble_auc"),
            "PR_AUC": metrics.get("ensemble_pr_auc"),
            "ACC": metrics.get("ensemble_accuracy"),
            "top15_mean_excess_return": metrics.get("top15_mean_excess_return"),
            "top15_hit_rate": metrics.get("top15_hit_rate"),
            "lgbm_classifier_auc": metrics.get("lgbm_classifier_auc"),
            "rank_spearman": metrics.get("rank_spearman"),
            "ndcg_at_10": metrics.get("ndcg_at_10"),
        },
        "top_selection_count": len(artifact.get("top_selection", [])),
        "notes": [
            "Train period: 24 months",
            "Validation period: 3 months",
            "Walk-forward step: 3 months",
            "Target horizon: 63 trading days (approx. 3 months)",
            "Final score: 0.70 * classifier_probability + 0.30 * rank_score",
        ],
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def train_and_save_model() -> dict[str, Any]:
    artifact = train_market_ensemble()
    print(artifact["metadata"])
    print("Saved:", artifact["model_path"])
    return artifact


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    train_and_save_model()
