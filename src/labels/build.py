"""Label generation for return-based targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_utc_index


def build_labels(candles: pd.DataFrame, horizon_name: str, steps_ahead: int = 1) -> pd.DataFrame:
    if steps_ahead <= 0:
        raise ValueError("steps_ahead must be >= 1")
    if candles.empty:
        return pd.DataFrame(columns=["ts_utc", f"target_r_{horizon_name}"])

    frame = ensure_utc_index(candles)
    close = frame["close"]
    future_close = close.shift(-steps_ahead)
    target = np.log(future_close / close)
    out = pd.DataFrame(
        {
            "ts_utc": frame["ts_utc"],
            f"target_r_{horizon_name}": target,
        }
    )
    out = out.dropna().reset_index(drop=True)
    return out


def join_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    merged = features.merge(labels, on="ts_utc", how="inner")
    merged = merged.dropna(subset=[label_col]).reset_index(drop=True)
    return merged

