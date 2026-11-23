"""Feature engineering on 4h candles.

Every feature at row t uses only candles that closed at or before the close of bar t.
The target at row t is whether the *next* bar's log return is positive.

Follows ``apply_feature_engineering`` from retail-sales-forecasting (target lags plus
calendar features) with one change: rows with NaN are dropped, never filled with 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RET_LAGS = (1, 2, 3, 6, 12, 42)  # 42 bars = 1 week, the counterpart of Lag 52 (weeks)
VOL_WINDOWS = (6, 42)
RSI_PERIOD = 14
EMA_FAST, EMA_SLOW = 12, 26
VOLZ_WINDOW = 42
# Longest look-back of any rolling-window feature: rows whose window touches an
# empty bar are dropped (EMA-based features would otherwise silently bridge gaps).
GAP_LOOKBACK = max(max(RET_LAGS), max(VOL_WINDOWS), VOLZ_WINDOW, RSI_PERIOD, EMA_SLOW)

FEATURES = (
    [f"ret_lag_{k}" for k in RET_LAGS]
    + [f"vol_{w}" for w in VOL_WINDOWS]
    + ["rsi_14", "ema_diff", "range_close", "volume_z", "hour", "day_of_week"]
)
TARGET = "target"
NEXT_RET = "next_ret"  # simple return of the next bar, used by the portfolio


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI (exponential smoothing with alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss > 0, 100.0).where(avg_gain.notna())


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with FEATURES, TARGET, NEXT_RET and helper columns.

    ``bars`` must be on a regular 4h grid (see ``data.resample_4h``); empty bars are NaN.
    NaN rows are *not* dropped here; ``dataset`` does that.
    """
    close = bars["close"]
    log_close = np.log(close)
    ret = log_close.diff()  # log return of bar t (known at its close)

    out = pd.DataFrame(index=bars.index)
    for k in RET_LAGS:
        # ret_lag_k = the target k bars ago: lag 1 is the return of bar t itself
        out[f"ret_lag_{k}"] = ret.shift(k - 1)
    for w in VOL_WINDOWS:
        out[f"vol_{w}"] = ret.rolling(w, min_periods=w).std()
    out["rsi_14"] = _rsi(close, RSI_PERIOD)
    ema_fast = close.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
    out["ema_diff"] = np.log(ema_fast / ema_slow)
    out["range_close"] = (bars["high"] - bars["low"]) / close
    log_vol = np.log1p(bars["volume"])
    mu = log_vol.rolling(VOLZ_WINDOW, min_periods=VOLZ_WINDOW).mean()
    sd = log_vol.rolling(VOLZ_WINDOW, min_periods=VOLZ_WINDOW).std()
    out["volume_z"] = (log_vol - mu) / sd
    out["hour"] = bars.index.hour
    out["day_of_week"] = bars.index.dayofweek

    # Rows whose look-back window contains an empty bar are invalid
    empty = close.isna().astype(float)
    touched = empty.rolling(GAP_LOOKBACK + 1, min_periods=1).max() > 0
    out.loc[touched, FEATURES] = np.nan

    next_ret = ret.shift(-1)
    out[TARGET] = (next_ret > 0).astype(float).where(next_ret.notna())
    out[NEXT_RET] = np.expm1(next_ret)
    return out


def dataset(bars: pd.DataFrame) -> pd.DataFrame:
    """Features plus target with every NaN row dropped."""
    df = build_features(bars)
    df = df.dropna(subset=FEATURES + [TARGET, NEXT_RET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def latest_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Feature row(s) for the most recent closed bar (target unknown yet)."""
    return build_features(bars)[FEATURES].dropna().tail(1)
