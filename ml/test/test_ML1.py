import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURE_PANEL_PATH
from ml.features import FEATURE_COLUMNS
from ml.storage import read_frame


RANDOM_SEED = 42


# -----------------------------
# 1) 3개월 future target 만들기
# -----------------------------
def add_3m_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).copy()
    horizon = 63

    panel["future_return_3M"] = (
        panel.groupby("ticker")["Close"]
        .transform(lambda series: series.shift(-horizon) / series - 1)
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


# -----------------------------
# 2) rolling train/valid split
# -----------------------------
def make_rolling_folds(
    panel: pd.DataFrame,
    train_months: int = 36,
    valid_months: int = 3,
    step_months: int = 3,
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
        if not train.empty and not valid.empty:
            folds.append((train, valid))
    return folds


def transformer_block(inputs, *, tf, d_model: int, num_heads: int, dropout: float):
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(inputs, inputs)
    attention = tf.keras.layers.Dropout(dropout)(attention)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    feed_forward = tf.keras.layers.Dense(d_model * 2, activation="gelu")(x)
    feed_forward = tf.keras.layers.Dropout(dropout)(feed_forward)
    feed_forward = tf.keras.layers.Dense(d_model)(feed_forward)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + feed_forward)


def build_transformer_classifier(
    feature_count: int,
    *,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    dropout: float = 0.15,
) -> object:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required. Install project dependencies with `uv sync`."
        ) from error

    class FeatureTokenEmbedding(tf.keras.layers.Layer):
        """Turn each scalar tabular feature into one Transformer token."""

        def __init__(self):
            super().__init__()
            self.value_projection = tf.keras.layers.Dense(d_model)
            self.position_embedding = tf.keras.layers.Embedding(feature_count, d_model)

        def call(self, layer_inputs):
            values = self.value_projection(tf.expand_dims(layer_inputs, axis=-1))
            positions = self.position_embedding(tf.range(feature_count))
            return values + positions[tf.newaxis, :, :]

    inputs = tf.keras.Input(shape=(feature_count,), name="features")
    x = FeatureTokenEmbedding()(inputs)
    for _ in range(num_blocks):
        x = transformer_block(
            x, tf=tf, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)

    model = tf.keras.Model(inputs, outputs, name="tabular_transformer_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def balanced_class_weights(target: np.ndarray) -> dict[int, float]:
    counts = np.bincount(target.astype(int), minlength=2)
    total = counts.sum()
    return {
        class_id: total / (2.0 * count)
        for class_id, count in enumerate(counts)
        if count > 0
    }


# -----------------------------
# 3) TensorFlow Transformer 훈련/평가
# -----------------------------
def train_transformer_folds(
    panel: pd.DataFrame,
    *,
    epochs: int = 50,
    batch_size: int = 256,
):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required. Install project dependencies with `uv sync`."
        ) from error

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    panel = add_3m_targets(panel)
    feature_cols = [column for column in FEATURE_COLUMNS if column in panel.columns]
    if not feature_cols:
        raise ValueError("Feature columns not found in panel")

    fold_results = []
    for idx, (train, valid) in enumerate(make_rolling_folds(panel), 1):
        train_medians = train[feature_cols].median()
        X_train_frame = train[feature_cols].fillna(train_medians).fillna(0.0)
        X_valid_frame = valid[feature_cols].fillna(train_medians).fillna(0.0)
        y_train = train["target_3M"].to_numpy(dtype=np.int32)
        y_valid = valid["target_3M"].to_numpy(dtype=np.int32)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_frame).astype(np.float32)
        X_valid = scaler.transform(X_valid_frame).astype(np.float32)

        tf.keras.backend.clear_session()
        model = build_transformer_classifier(len(feature_cols))
        model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=balanced_class_weights(y_train),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=7,
                    restore_best_weights=True,
                )
            ],
            verbose=1,
        )

        probability = model.predict(X_valid, batch_size=batch_size, verbose=0).reshape(-1)
        prediction = (probability >= 0.5).astype(int)
        fold_results.append({
            "fold": idx,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "valid_start": valid["date"].min(),
            "valid_end": valid["date"].max(),
            "AUC": roc_auc_score(y_valid, probability),
            "PR_AUC": average_precision_score(y_valid, probability),
            "ACC": accuracy_score(y_valid, prediction),
        })

    return fold_results


# -----------------------------
# 4) 실행 예시
# -----------------------------
if __name__ == "__main__":
    feature_panel = read_frame(FEATURE_PANEL_PATH)
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    results = train_transformer_folds(feature_panel)

    for row in results:
        print(row)

    avg_auc = np.mean([result["AUC"] for result in results])
    avg_pr_auc = np.mean([result["PR_AUC"] for result in results])
    print(f"\nAverage AUC: {avg_auc:.4f}")
    print(f"Average PR-AUC: {avg_pr_auc:.4f}")
