"""Data cleaning and normalization."""

from __future__ import annotations

import pandas as pd

from src.utils import ensure_utc_index


def clean_ohlcv(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "exchange",
            ]
        )

    frame = raw_df.copy()
    frame = ensure_utc_index(frame, column="ts_utc")

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["ts_utc", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["ts_utc"], keep="last")
    frame = frame[required_cols + ["ts_utc", "exchange"]]
    frame["symbol"] = symbol
    frame = frame[["ts_utc", "open", "high", "low", "close", "volume", "symbol", "exchange"]]
    return frame.reset_index(drop=True)

