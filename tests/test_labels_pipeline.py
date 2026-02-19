from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import build_features, finalize_features
from src.labels.build import build_labels, join_features_labels


def _make_candles(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 50000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.uniform(50.0, 300.0, n)
    return pd.DataFrame(
        {
            "ts_utc": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "BTC/USDT",
            "exchange": "binance",
        }
    )


def test_label_shift_and_join_no_future_rows() -> None:
    candles = _make_candles()
    feature_cfg = {
        "return_windows": [1, 2, 3],
        "vol_windows": [5, 10],
        "volume_windows": [5],
        "trend_windows": [5, 10],
        "rsi_window": 5,
        "atr_window": 5,
        "macd": {"fast": 3, "slow": 6, "signal": 3},
    }
    features = finalize_features(build_features(candles, feature_cfg), warmup_drop=True)
    labels = build_labels(candles, "5m", steps_ahead=1)
    ds = join_features_labels(features, labels, "target_r_5m")

    assert len(ds) > 0
    # the last timestamp from original candles cannot be used because t+H is unknown
    assert ds["ts_utc"].max() < candles["ts_utc"].max()
    # no NaN after warm-up and join
    assert not ds.isna().any().any()
