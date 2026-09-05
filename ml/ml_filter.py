from typing import Any

import joblib
import pandas as pd

from analysis.technical import get_stock_data
from ml.features import FEATURE_COLUMNS
from ml.train_model import MODEL_PATH, MODEL_VERSION, resolve_stock, train_market_ensemble


def _load_current_artifact(latest_session: str) -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        artifact = joblib.load(MODEL_PATH)
        metadata = artifact["metadata"]
        valid = (
            metadata.get("model_version") == MODEL_VERSION
            and metadata.get("trained_through") == latest_session
            and metadata.get("feature_columns") == FEATURE_COLUMNS
            and set(artifact.get("models", {}))
            == {"lgbm_classifier", "lgbm_ranker"}
        )
        return artifact if valid else None
    except (KeyError, TypeError, ValueError, OSError):
        return None


def predict_up_probability(
    stock_name_or_ticker: str,
    df: pd.DataFrame | None = None,
    economy_df: pd.DataFrame | None = None,
) -> dict:
    del economy_df  # Economy data is trained and scored as part of the full panel.
    ticker, company_name = resolve_stock(stock_name_or_ticker)
    market_data = df if df is not None else get_stock_data(ticker, period="5y")
    latest_session = pd.Timestamp(market_data.index.max()).date().isoformat()
    artifact = _load_current_artifact(latest_session)
    model_reused = artifact is not None
    if artifact is None:
        artifact = train_market_ensemble()

    prediction = artifact["latest_predictions"].get(ticker)
    if prediction is None:
        raise ValueError(
            f"{ticker} is not in the trained KOSPI/KOSDAQ top-200 universe"
        )
    score = float(prediction["ml_score"])
    probability = float(prediction.get("classification_probability", prediction.get("lgbm_probability", score)))
    ml_rank = int(prediction.get("ml_rank", 9999))
    metadata = artifact["metadata"]
    return {
        "ticker": ticker,
        "company_name": company_name,
        "up_probability": probability,
        "classification_probability": probability,
        "ml_score": score,
        "ml_rank": ml_rank,
        "lgbm_probability": float(prediction["lgbm_probability"]),
        "return_rank": float(prediction["return_rank"]),
        "ml_pass": ml_rank <= 10,
        "model_reused": model_reused,
        "model_path": str(MODEL_PATH),
        "trained_through": metadata["trained_through"],
        "algorithm": metadata["algorithm"],
        "weights": metadata["weights"],
        "metrics": metadata["metrics"],
    }
