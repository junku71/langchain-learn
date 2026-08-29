from pathlib import Path

import pandas as pd

from stock_analyzer import (
    get_stock_data,
    calculate_indicators,
)

# ------------------------------------
# feature 생성
#-----------------------------------

def create_ml_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df = calculate_indicators(df)

    df["MA5_DISTANCE"] = (
        df["Close"] - df["MA5"]
    ) / df["MA5"]

    df["MA20_DISTANCE"] = (
        df["Close"] - df["MA20"]
    ) / df["MA20"]

    df["ATR_RATIO"] = (
        df["ATR"] / df["Close"]
    )

    return df

# ------------------------------------
# 타겟 만들기
# -----------------------------------
def create_target(
    df: pd.DataFrame,
    horizon: int = 10
):

    df = df.copy()

    df["FUTURE_CLOSE"] = (
        df["Close"]
        .shift(-horizon)
    )

    df["TARGET"] = (
        df["FUTURE_CLOSE"]
        > df["Close"]
    ).astype(int)

    return df

FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "ADX",
    "DI_PLUS",
    "DI_MINUS",
    "VOLUME_RATIO",
    "MA5_DISTANCE",
    "MA20_DISTANCE",
    "ATR_RATIO",
]

# ------------------------------------
# ML model: Logistic Regression
# -----------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# 학습

df = get_stock_data(
    "005930.KS",
    period="5y"
)

df = create_ml_features(df)

df = create_target(
    df,
    horizon=10
)

df = df.dropna()

X = df[
    FEATURE_COLUMNS
]

y = df[
    "TARGET"
]


# Keep chronological order to avoid leaking future prices into training data.

split_index = int(
    len(df) * 0.8
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]

# model 학습
model.fit(
    X_train,
    y_train
)

# model 평가
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

pred = model.predict(
    X_test
)

prob = model.predict_proba(
    X_test
)[:, 1]

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print(
    "AUC:",
    roc_auc_score(
        y_test,
        prob
    )
)


import joblib
model_path = Path("ml/stock_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    model_path
)
