from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "abs_return",
    "intraday_range",
    "close_open_gap",
    "volume_change",
    "momentum_5",
    "vol_5",
    "vol_20",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features using information available at or before day t."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    price = out["adj_close"].astype(float)
    out["log_return"] = np.log(price).diff()
    out["abs_return"] = out["log_return"].abs()
    out["intraday_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["close_open_gap"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["volume_change"] = np.log1p(out["volume"]).diff()
    out["momentum_5"] = np.log(price / price.shift(5))
    out["vol_5"] = out["log_return"].rolling(5).std(ddof=1) * np.sqrt(252)
    out["vol_20"] = out["log_return"].rolling(20).std(ddof=1) * np.sqrt(252)
    return out


def forward_realized_vol(log_returns: pd.Series, horizon: int = 5) -> pd.Series:
    """Annualized realized volatility over t+1 ... t+horizon."""
    values = log_returns.to_numpy(dtype=float)
    target = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values) - horizon):
        window = values[i + 1 : i + 1 + horizon]
        if np.isfinite(window).all() and len(window) >= 2:
            target[i] = float(np.std(window, ddof=1) * np.sqrt(252))
    return pd.Series(target, index=log_returns.index, name="target_vol")


def ewma_volatility(log_returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA annualized volatility using information through t."""
    r = log_returns.fillna(0.0).to_numpy(dtype=float)
    var = np.zeros_like(r)
    if len(r) == 0:
        return pd.Series(dtype=float)
    var[0] = r[0] ** 2
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i] ** 2
    return pd.Series(np.sqrt(np.maximum(var, 0.0) * 252), index=log_returns.index, name="ewma")


def prepare_feature_frame(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = build_features(df)
    out["target_vol"] = forward_realized_vol(out["log_return"], horizon=horizon)
    out["ewma"] = ewma_volatility(out["log_return"])
    return out
