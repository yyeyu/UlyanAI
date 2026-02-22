"""Deterministic OHLCV resampling."""

from __future__ import annotations

import pandas as pd

from src.utils import TIMEFRAME_TO_PANDAS_RULE


def resample_ohlcv(clean_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if clean_df.empty:
        return clean_df.copy()
    if timeframe not in TIMEFRAME_TO_PANDAS_RULE:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    rule = TIMEFRAME_TO_PANDAS_RULE[timeframe]
    frame = clean_df.copy()
    frame = frame.set_index("ts_utc").sort_index()

    # Right-closed/right-labeled buckets keep alignment with candle close moments.
    agg = frame.resample(rule, label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        symbol=("symbol", "last"),
        exchange=("exchange", "last"),
    )

    # Drop buckets that are labeled beyond the last available source candle.
    agg = agg[agg.index <= frame.index.max()]
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg = agg.reset_index()
    return agg
