from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, prepare_feature_frame
from .model import TrainingResult, predict, train_lstm


@dataclass
class PipelineResult:
    feature_frame: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    next_forecast: float
    scaler: StandardScaler
    training: TrainingResult


def _sequence_positions(frame: pd.DataFrame, lookback: int) -> list[int]:
    usable = frame["target_vol"].notna().to_numpy()
    feature_ok = frame[FEATURE_COLUMNS].notna().all(axis=1).to_numpy()
    positions = []
    for i in range(lookback - 1, len(frame)):
        window_ok = feature_ok[i - lookback + 1 : i + 1].all()
        if window_ok and usable[i]:
            positions.append(i)
    return positions


def _make_sequences(
    scaled_features: np.ndarray,
    targets: np.ndarray,
    positions: list[int],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([scaled_features[i - lookback + 1 : i + 1] for i in positions])
    y = np.asarray([targets[i] for i in positions], dtype=float)
    return X.astype(np.float32), y.astype(np.float32)


def _metric_rows(y_true: np.ndarray, pred: np.ndarray, name: str) -> dict[str, float | str]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred)) if len(y_true) > 1 else float("nan")
    corr = float(np.corrcoef(y_true, pred)[0, 1]) if len(y_true) > 2 and np.std(pred) > 0 else float("nan")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2, "Correlation": corr}


def run_pipeline(
    market_df: pd.DataFrame,
    lookback: int = 20,
    horizon: int = 5,
    epochs: int = 25,
) -> PipelineResult:
    if lookback < 5:
        raise ValueError("lookback must be at least 5 trading days")
    if horizon < 2:
        raise ValueError("horizon must be at least 2 trading days")

    frame = prepare_feature_frame(market_df, horizon=horizon).reset_index(drop=True)
    positions = _sequence_positions(frame, lookback=lookback)
    if len(positions) < 120:
        raise RuntimeError(
            f"Only {len(positions)} supervised sequences are available; at least 120 are required. "
            "Use the full Yahoo history rather than the short fallback dataset when possible."
        )

    n = len(positions)
    train_end = max(int(n * 0.70), 1)
    val_end = max(int(n * 0.85), train_end + 1)
    if val_end >= n:
        val_end = n - 1

    train_positions = positions[:train_end]
    val_positions = positions[train_end:val_end]
    test_positions = positions[val_end:]

    train_cut = train_positions[-1]
    scaler = StandardScaler()
    train_feature_rows = frame.loc[:train_cut, FEATURE_COLUMNS].dropna()
    scaler.fit(train_feature_rows.to_numpy(dtype=float))

    feature_values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    finite_mask = np.isfinite(feature_values)
    safe_values = np.where(finite_mask, feature_values, 0.0)
    scaled = scaler.transform(safe_values)
    targets = frame["target_vol"].to_numpy(dtype=float)

    X_train, y_train = _make_sequences(scaled, targets, train_positions, lookback)
    X_val, y_val = _make_sequences(scaled, targets, val_positions, lookback)
    X_test, y_test = _make_sequences(scaled, targets, test_positions, lookback)

    training = train_lstm(X_train, y_train, X_val, y_val, epochs=epochs)
    lstm_pred = predict(training.model, X_test)
    ewma_pred = frame.loc[test_positions, "ewma"].to_numpy(dtype=float)

    pred_df = pd.DataFrame(
        {
            "date": frame.loc[test_positions, "date"].to_numpy(),
            "actual_vol": y_test.astype(float),
            "lstm_vol": lstm_pred,
            "ewma_vol": ewma_pred,
        }
    )

    metrics = pd.DataFrame(
        [
            _metric_rows(y_test, lstm_pred, "LSTM"),
            _metric_rows(y_test, ewma_pred, "EWMA baseline"),
        ]
    )

    latest_features = frame[FEATURE_COLUMNS].dropna().tail(lookback)
    if len(latest_features) < lookback:
        raise RuntimeError("Not enough recent rows for the next forecast.")
    latest_scaled = scaler.transform(latest_features.to_numpy(dtype=float))[None, :, :].astype(np.float32)
    next_forecast = float(predict(training.model, latest_scaled)[0])

    return PipelineResult(frame, pred_df, metrics, next_forecast, scaler, training)
