# lstm_local_weather.py
# ------------------------------------------------------------
# Quick start (in a terminal / cmd):
#   1) cd C:\Users\Nemasha\Desktop\lstm
#   2) py -m pip install pandas numpy scikit-learn tensorflow joblib matplotlib
#   3) py lstm_local_weather.py --csv "C:\Users\Nemasha\Desktop\lstm\colombo 2022-11-15 to 2025-08-10.csv"
# Optional flags:
#   --epochs 50 --batch 32 --model_dir "local_weather_model"
# ------------------------------------------------------------

import os
import argparse
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt


class WeatherForecaster:
    def __init__(self, sequence_length=30, forecast_horizon=7, learning_rate=0.001):
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = [
            'temp', 'humidity', 'precip', 'windspeed', 'winddir',
            'cloudcover', 'dew', 'uvindex', 'sealevelpressure'
        ]
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.all_features = []
        self.learning_rate = learning_rate

    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the weather data from CSV path"""
        last_err = None
        for enc in (None, "utf-8", "utf-8-sig", "ISO-8859-1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception as e:
                last_err = e
        if 'df' not in locals():
            raise RuntimeError(f"Failed to read CSV. Last error: {last_err}")

        if 'datetime' not in df.columns:
            raise ValueError("CSV must contain a 'datetime' column.")

        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV is missing required columns: {missing_cols}")

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        features_df = df[['datetime'] + self.feature_columns].copy()
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        # date features
        features_df['day_of_year'] = features_df['datetime'].dt.dayofyear
        features_df['month'] = features_df['datetime'].dt.month
        features_df['day_of_week'] = features_df['datetime'].dt.dayofweek

        # cyclical encodings
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)

        self.all_features = self.feature_columns + [
            'day_of_year', 'month', 'day_of_week',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        return features_df

    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        out_dim = len(self.feature_columns)  # predict only original 9 features
        for i in range(self.sequence_length, len(data) - self.forecast_horizon + 1):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i:i + self.forecast_horizon, :out_dim])
        return np.array(X), np.array(y)

    def build_model(self, input_shape, output_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse',
                      metrics=['mae'])
        return model

    def train(self, csv_path, epochs=50, batch_size=32, model_dir="local_weather_model"):
        """Train the weather forecasting model"""
        print("\n--- Starting Data Preprocessing ---")
        features_df = self.load_and_preprocess_data(csv_path)

        feature_data = features_df[self.all_features].values
        scaled_data = self.scaler.fit_transform(feature_data)

        print("Creating sequences...")
        X, y = self.create_sequences(scaled_data)
        if len(X) == 0:
            raise ValueError(
                "Not enough rows for the chosen sequence_length and forecast_horizon. "
                f"Need at least {self.sequence_length + self.forecast_horizon} rows."
            )

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        # Flatten y across the horizon * features for training
        y_flat = y.reshape(y.shape[0], -1)

        # Chronological split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_flat[:split_idx], y_flat[split_idx:]
        y_test_unflat = y[split_idx:]  # keep an unflattened copy for metrics later

        print("\n--- Building and Training Model ---")
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = y_train.shape[1]
        self.model = self.build_model(input_shape, output_shape)

        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir, "best_weights.keras")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks
        )

        print("\n--- Evaluating Model ---")
        y_pred_flat = self.model.predict(X_test)
        y_pred = y_pred_flat.reshape(y_test_unflat.shape)  # (samples, horizon, features)

        metrics = {}
        out_dim = len(self.feature_columns)
        for i, feature in enumerate(self.feature_columns):
            mae = mean_absolute_error(y_test_unflat[:, :, i].ravel(), y_pred[:, :, i].ravel())
            rmse = np.sqrt(mean_squared_error(y_test_unflat[:, :, i].ravel(), y_pred[:, :, i].ravel()))
            metrics[feature] = {'mae': float(mae), 'rmse': float(rmse)}
            print(f"{feature}: MAE={mae:.3f}, RMSE={rmse:.3f}")

        # return test arrays for plotting
        return history, metrics, features_df, y_test_unflat, y_pred

    def predict_next_7_days(self, features_df):
        """Predict weather for the next 7 days using the latest sequence in features_df"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        recent_data = features_df[self.all_features].tail(self.sequence_length).values
        recent_scaled = self.scaler.transform(recent_data)
        X_pred = recent_scaled.reshape(1, self.sequence_length, -1)

        pred_scaled_flat = self.model.predict(X_pred)
        pred_scaled = pred_scaled_flat.reshape(self.forecast_horizon, len(self.feature_columns))

        # Build a dummy array that includes all features so we can inverse_transform
        dummy = np.zeros((self.forecast_horizon, len(self.all_features)))
        dummy[:, :len(self.feature_columns)] = pred_scaled
        pred_unscaled_full = self.scaler.inverse_transform(dummy)
        pred_unscaled = pred_unscaled_full[:, :len(self.feature_columns)]

        base_date = pd.to_datetime(features_df['datetime'].iloc[-1])
        forecast_dates = [base_date + timedelta(days=i + 1) for i in range(self.forecast_horizon)]

        forecast_df = pd.DataFrame(pred_unscaled, columns=self.feature_columns)
        forecast_df['date'] = [d.strftime('%Y-%m-%d') for d in forecast_dates]
        cols = ['date'] + self.feature_columns
        return forecast_df[cols]

    def save_model(self, model_dir='local_weather_model'):
        """Save the trained model (.h5) and scaler (.joblib)"""
        if self.model is None:
            raise ValueError("No model to save!")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")


# ----------------------- Plotting helpers -----------------------

def plot_training_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # MAE (if present)
    if "mae" in history.history:
        plt.figure()
        plt.plot(history.history["mae"], label="train")
        if "val_mae" in history.history:
            plt.plot(history.history["val_mae"], label="val")
        plt.title("Training MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_mae.png"))
        plt.close()


def plot_forecast(forecast_df, out_dir, feature_columns):
    os.makedirs(out_dir, exist_ok=True)
    dates = pd.to_datetime(forecast_df["date"])

    # One chart per feature
    for col in feature_columns:
        plt.figure()
        plt.plot(dates, forecast_df[col])
        plt.title(f"7-Day Forecast — {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.tight_layout()
        fname = f"forecast_{col}.png".replace("/", "_")
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    # Also save a combined CSV for convenience
    forecast_df.to_csv(os.path.join(out_dir, "forecast_7day.csv"), index=False)


def plot_actual_vs_pred(y_true, y_pred, feature_columns, out_dir):
    """
    One image per feature comparing Actual vs Predicted over the whole test set.
    y_true, y_pred: arrays with shape (samples, horizon, features)
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, col in enumerate(feature_columns):
        plt.figure()
        plt.plot(y_true[:, :, i].flatten(), label="Actual")
        plt.plot(y_pred[:, :, i].flatten(), label="Predicted")
        plt.title(f"Actual vs Predicted — {col} (Test Set)")
        plt.xlabel("Time steps in test set")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        fname = f"actual_vs_pred_{col}.png".replace("/", "_")
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()


def plot_all_features_actual_vs_pred(y_true, y_pred, feature_columns, out_dir):
    """
    Plot ALL features' actual and predicted values on a single graph.
    y_true, y_pred: shape (samples, horizon, features)
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))

    for i, col in enumerate(feature_columns):
        plt.plot(y_true[:, :, i].flatten(), label=f"Actual {col}")
        plt.plot(y_pred[:, :, i].flatten(), '--', label=f"Pred {col}")

    plt.title("Actual vs Predicted — All Features (Test Set)")
    plt.xlabel("Time steps in test set")
    plt.ylabel("Value")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "actual_vs_pred_all_features.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LSTM weather model and forecast next 7 days.")
    parser.add_argument("--csv", required=False, default="", help="Path to input CSV file.")
    parser.add_argument("--sequence", type=int, default=30, help="Sequence length (default 30).")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days (default 7).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default 50).")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default 32).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default 0.001).")
    parser.add_argument("--model_dir", default="local_weather_model", help="Where to save model, scaler & plots.")
    args = parser.parse_args()

    csv_path = args.csv.strip()
    if not csv_path:
        csv_path = input("Enter full path to CSV file: ").strip('"').strip()

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print("\n--- Starting Weather Forecast Process ---")
    print(f"CSV: {csv_path}")

    forecaster = WeatherForecaster(
        sequence_length=args.sequence,
        forecast_horizon=args.horizon,
        learning_rate=args.lr
    )

    print("\nTraining model...")
    history, metrics, features_df, y_test_unflat, y_pred = forecaster.train(
        csv_path=csv_path,
        epochs=args.epochs,
        batch_size=args.batch,
        model_dir=args.model_dir
    )

    # Plot training curves
    plot_training_history(history, args.model_dir)

    # Plot per-feature and all-feature actual vs predicted
    plot_actual_vs_pred(y_test_unflat, y_pred, forecaster.feature_columns, args.model_dir)
    plot_all_features_actual_vs_pred(y_test_unflat, y_pred, forecaster.feature_columns, args.model_dir)

    print("\nSaving model...")
    forecaster.save_model(args.model_dir)

    print("\nGenerating forecast...")
    forecast = forecaster.predict_next_7_days(features_df)
    print("\n--- 7-Day Weather Forecast ---")
    print(forecast.round(2).to_string(index=False))

    # Save forecast plots + CSV into the model_dir
    plot_forecast(forecast, args.model_dir, forecaster.feature_columns)
    print(f"\n✅ Plots saved in: {args.model_dir}")
    print(f"   - training_loss.png")
    print(f"   - training_mae.png (if MAE tracked)")
    print(f"   - actual_vs_pred_<feature>.png for each feature")
    print(f"   - actual_vs_pred_all_features.png")
    print(f"   - forecast_<feature>.png for each feature")
    print(f"   - forecast_7day.csv")
    print("\nForecast completed successfully!")


if __name__ == "__main__":
    main()
