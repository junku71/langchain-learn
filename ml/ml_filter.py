import joblib
import pandas as pd

from ml.train_model import (
    create_ml_features,
    FEATURE_COLUMNS,
)

model = joblib.load(
    "ml/stock_model.pkl"
)

# ------------------------------------
# 예측함수, 65% 이상이면 매수 추천
#-----------------------------------


def predict_up_probability(
    df: pd.DataFrame
) -> dict:

    feature_df = create_ml_features(
        df
    )

    latest = feature_df[
        FEATURE_COLUMNS
    ].iloc[[-1]]

    probability = (
        model.predict_proba(
            latest
        )[0][1]
    )

    return {
        "up_probability":
            float(probability),

        "ml_pass":
            probability >= 0.65
    }