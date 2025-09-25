"""End-to-end training utilities for LSTM, Random Forest, SVR, SARIMA, and ARIMA.

All models operate on the nine core features expected by the forecasting app:
`temp`, `humidity`, `precip`, `windspeed`, `winddir`, `cloudcover`, `dew`,
`uvindex`, `sealevelpressure`.

Usage (from repo root):

    python -m models.train_all_models --csv data/colombo22-25.csv

This script performs the following steps:
1. Load & sort the dataset by datetime
2. Engineer cyclical date features (day-of-year, month, weekday)
3. Build 30-day input sequences with a 7-day forecast horizon
4. Split chronologically into train/test
5. Fit each model and report MAE/RMSE/R^2 on the hold-out horizon

The code is designed to be a reproducible baseline; hyper-parameters can be
adjusted via CLI flags.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

FEATURE_COLUMNS = [
    "temp",
    "humidity",
    "precip",
    "windspeed",
    "winddir",
    "cloudcover",
    "dew",
    "uvindex",
    "sealevelpressure",
]

CYCLICAL_COLUMNS = [
    "day_of_year",
    "month",
    "day_of_week",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

@dataclass
class SequenceDataset:
    X_seq: np.ndarray  # shape (samples, seq_len, n_features)
    y_seq: np.ndarray  # shape (samples, horizon, n_targets)
    feature_scaler: MinMaxScaler


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("CSV must include a 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_year"] = out["datetime"].dt.dayofyear
    out["month"] = out["datetime"].dt.month
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    feature_cols: List[str],
) -> SequenceDataset:
    all_cols = feature_cols + CYCLICAL_COLUMNS
    data = df[all_cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i in range(seq_len, len(df) - horizon + 1):
        X_list.append(scaled[i - seq_len : i, :])
        y_slice = scaled[i : i + horizon, : len(feature_cols)]
        y_list.append(y_slice)

    X_seq = np.stack(X_list)
    y_seq = np.stack(y_list)
    return SequenceDataset(X_seq=X_seq, y_seq=y_seq, feature_scaler=scaler)


def train_test_split_sequences(dataset: SequenceDataset, train_ratio: float) -> Tuple:
    n_samples = dataset.X_seq.shape[0]
    train_end = max(1, int(n_samples * train_ratio))
    X_train = dataset.X_seq[:train_end]
    y_train = dataset.y_seq[:train_end]
    X_test = dataset.X_seq[train_end:]
    y_test = dataset.y_seq[train_end:]
    return X_train, X_test, y_train, y_test


def inverse_transform_predictions(
    preds_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    horizon, n_targets = preds_scaled.shape
    dummy = np.zeros((horizon, scaler.n_features_in_), dtype=np.float32)
    dummy[:, :n_targets] = preds_scaled
    unscaled = scaler.inverse_transform(dummy)
    return unscaled[:, :n_targets]


def unscale_sequence_batch(batch: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return np.stack([inverse_transform_predictions(sample, scaler) for sample in batch])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_lstm(X_train, y_train, seq_len, n_features, horizon, lr, epochs, batch):
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dense(horizon * len(FEATURE_COLUMNS), activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    model.fit(
        X_train,
        y_train.reshape(len(y_train), -1),
        epochs=epochs,
        batch_size=batch,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )
    return model


def predict_lstm(model, X_test, horizon):
    preds = model.predict(X_test, verbose=0)
    return preds.reshape(len(X_test), horizon, len(FEATURE_COLUMNS))


def train_random_forest(X_train, y_train):
    X_flat = X_train.reshape(len(X_train), -1)
    y_flat = y_train.reshape(len(y_train), -1)
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
    rf.fit(X_flat, y_flat)
    return rf


def predict_random_forest(model, X_test, horizon):
    preds = model.predict(X_test.reshape(len(X_test), -1))
    return preds.reshape(len(X_test), horizon, len(FEATURE_COLUMNS))


def train_svr(X_train, y_train):
    X_flat = X_train.reshape(len(X_train), -1)
    y_flat = y_train.reshape(len(y_train), -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    svr = MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
    svr.fit(X_scaled, y_flat)
    return svr, scaler


def predict_svr(model_tuple, X_test, horizon):
    svr, scaler = model_tuple
    X_flat = X_test.reshape(len(X_test), -1)
    X_scaled = scaler.transform(X_flat)
    preds = svr.predict(X_scaled)
    return preds.reshape(len(X_test), horizon, len(FEATURE_COLUMNS))


def train_stat_model(series: pd.Series, horizon: int, *, arima_order=None, seasonal_order=None):
    if seasonal_order is not None:
        model = SARIMAX(series, order=arima_order or (1, 0, 1), seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(series, order=arima_order or (2, 1, 2))
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return forecast


def evaluate_stat_models(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int, *, seasonal=False):
    preds = []
    actuals = []
    for feature in FEATURE_COLUMNS:
        series = train_df[feature]
        try:
            if seasonal:
                fc = train_stat_model(series, horizon, arima_order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
            else:
                fc = train_stat_model(series, horizon, arima_order=(2, 1, 2))
        except Exception:
            # fallback: repeat last value if model fails to converge
            fc = pd.Series([series.iloc[-1]] * horizon)
        preds.append(fc.to_numpy())
        actuals.append(test_df[feature].iloc[:horizon].to_numpy())
    preds = np.stack(preds, axis=1)  # shape (horizon, features)
    actuals = np.stack(actuals, axis=1)
    metrics = evaluate(actuals.reshape(-1), preds.reshape(-1))
    return metrics, preds, actuals


def main():
    parser = argparse.ArgumentParser(description="Train multiple forecasting models on weather features")
    parser.add_argument("--csv", default="data/colombo22-25.csv", help="Path to dataset with datetime column")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    df = load_dataset(Path(args.csv))
    df = engineer_features(df)

    dataset = build_sequences(df, seq_len=args.seq_len, horizon=args.horizon, feature_cols=FEATURE_COLUMNS)
    X_train, X_test, y_train, y_test = train_test_split_sequences(dataset, args.train_ratio)

    if len(X_test) == 0:
        raise RuntimeError("Not enough samples for the chosen split. Reduce sequence length or horizon.")

    y_train_unscaled = unscale_sequence_batch(y_train, dataset.feature_scaler)
    y_test_unscaled = unscale_sequence_batch(y_test, dataset.feature_scaler)

    # Train LSTM
    lstm_model = train_lstm(
        X_train,
        y_train,
        seq_len=args.seq_len,
        n_features=dataset.X_seq.shape[-1],
        horizon=args.horizon,
        lr=args.lr,
        epochs=args.epochs,
        batch=args.batch,
    )
    lstm_preds_scaled = predict_lstm(lstm_model, X_test, args.horizon)
    lstm_preds = unscale_sequence_batch(lstm_preds_scaled, dataset.feature_scaler)
    lstm_metrics = evaluate(y_test_unscaled.reshape(-1), lstm_preds.reshape(-1))

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_preds_scaled = predict_random_forest(rf_model, X_test, args.horizon)
    rf_preds = unscale_sequence_batch(rf_preds_scaled, dataset.feature_scaler)
    rf_metrics = evaluate(y_test_unscaled.reshape(-1), rf_preds.reshape(-1))

    # SVR
    svr_model = train_svr(X_train, y_train)
    svr_preds_scaled = predict_svr(svr_model, X_test, args.horizon)
    svr_preds = unscale_sequence_batch(svr_preds_scaled, dataset.feature_scaler)
    svr_metrics = evaluate(y_test_unscaled.reshape(-1), svr_preds.reshape(-1))

    # ARIMA
    split_idx = args.seq_len + int(len(df) * args.train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    arima_metrics, _, _ = evaluate_stat_models(train_df, test_df, args.horizon, seasonal=False)

    # SARIMA
    sarima_metrics, _, _ = evaluate_stat_models(train_df, test_df, args.horizon, seasonal=True)

    print("\n=== Evaluation (flattened horizon) ===")
    for name, metrics in [
        ("LSTM", lstm_metrics),
        ("RandomForest", rf_metrics),
        ("SVR", svr_metrics),
        ("ARIMA", arima_metrics),
        ("SARIMA", sarima_metrics),
    ]:
        print(f"{name:12s} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
