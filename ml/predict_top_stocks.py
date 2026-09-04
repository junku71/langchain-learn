from __future__ import annotations

import argparse

import joblib
import pandas as pd

from ml.config import TOP_SELECTION_SIZE_PER_MARKET
from ml.features import FEATURE_COLUMNS
from ml.config import FEATURE_PANEL_PATH
from ml.storage import read_frame
from ml.train_model import MODEL_PATH, MODEL_VERSION, dataset_fingerprint, train_market_ensemble


def _artifact_is_current(artifact: dict) -> bool:
    try:
        metadata = artifact["metadata"]
        panel = read_frame(FEATURE_PANEL_PATH)
        return (
            not panel.empty
            and metadata.get("model_version") == MODEL_VERSION
            and metadata.get("feature_columns") == FEATURE_COLUMNS
            and metadata.get("trained_through") == str(pd.to_datetime(panel["date"]).max().date())
            and metadata.get("dataset_fingerprint") == dataset_fingerprint(panel)
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def predict_top_stocks(
    top: int = TOP_SELECTION_SIZE_PER_MARKET,
    retrain: bool = False,
) -> pd.DataFrame:
    artifact = None if retrain or not MODEL_PATH.exists() else joblib.load(MODEL_PATH)
    if artifact is not None and not _artifact_is_current(artifact):
        artifact = None
    if artifact is None:
        artifact = train_market_ensemble()
    result = pd.DataFrame.from_dict(artifact["latest_predictions"], orient="index")
    result.index.name = "ticker"
    ranked = result.reset_index().sort_values("ml_score", ascending=False)
    if "market" not in ranked:
        return ranked.head(top)
    return ranked.groupby("market", group_keys=False).head(top).sort_values(
        ["market", "ml_score"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict top Korean stocks")
    parser.add_argument(
        "--top", type=int, default=TOP_SELECTION_SIZE_PER_MARKET,
        help="Number of selections per market",
    )
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    print(predict_top_stocks(args.top, args.retrain).to_string(index=False))


if __name__ == "__main__":
    main()
