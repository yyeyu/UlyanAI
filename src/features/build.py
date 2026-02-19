"""Feature set construction for OHLCV candles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_utc_index


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def build_features(candles: pd.DataFrame, feature_cfg: dict) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame()
    frame = ensure_utc_index(candles)

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    volume = frame["volume"]

    out = frame[["ts_utc", "open", "high", "low", "close", "volume", "symbol", "exchange"]].copy()
    n_rows = len(frame)
    max_window = max(5, int(n_rows * 0.4))

    for k in feature_cfg.get("return_windows", [1, 2, 3, 6, 12]):
        if int(k) <= 0 or int(k) >= n_rows:
            continue
        out[f"log_return_{k}"] = np.log(close / close.shift(k))
        out[f"return_{k}"] = close.pct_change(k)

    if "log_return_1" not in out.columns:
        out["log_return_1"] = np.log(close / close.shift(1))
    log_return_1 = out["log_return_1"]
    for w in feature_cfg.get("vol_windows", [10, 20, 50]):
        if int(w) <= 1 or int(w) >= n_rows or int(w) > max_window:
            continue
        out[f"vol_std_{w}"] = log_return_1.rolling(window=w).std()

    out["candle_body"] = close - open_
    out["candle_range"] = high - low
    out["upper_wick"] = high - np.maximum(open_, close)
    out["lower_wick"] = np.minimum(open_, close) - low

    for w in feature_cfg.get("volume_windows", [20, 50]):
        if int(w) <= 1 or int(w) >= n_rows or int(w) > max_window:
            continue
        vol_mean = volume.rolling(window=w).mean()
        vol_std = volume.rolling(window=w).std()
        out[f"volume_mean_{w}"] = vol_mean
        out[f"volume_zscore_{w}"] = ((volume - vol_mean) / vol_std.replace(0, np.nan)).fillna(0.0)

    for w in feature_cfg.get("trend_windows", [20, 50, 200]):
        if int(w) <= 1 or int(w) >= n_rows or int(w) > max_window:
            continue
        ema = _ema(close, w)
        sma = close.rolling(window=w).mean()
        out[f"ema_{w}"] = ema
        out[f"sma_{w}"] = sma
        out[f"price_minus_ema_{w}"] = close - ema
        out[f"slope_ema_{w}"] = ema.diff()

    rsi_window = int(feature_cfg.get("rsi_window", 14))
    atr_window = int(feature_cfg.get("atr_window", 14))
    out["rsi"] = _rsi(close, min(max(rsi_window, 2), max(2, n_rows - 1)))
    out["atr"] = _atr(high, low, close, min(max(atr_window, 2), max(2, n_rows - 1)))

    macd_cfg = feature_cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    ema_fast = _ema(close, int(macd_cfg.get("fast", 12)))
    ema_slow = _ema(close, int(macd_cfg.get("slow", 26)))
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, int(macd_cfg.get("signal", 9)))
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_line - signal_line

    return out


def finalize_features(features: pd.DataFrame, warmup_drop: bool = True) -> pd.DataFrame:
    out = features.copy()
    if warmup_drop:
        out = out.dropna().reset_index(drop=True)
    return out
