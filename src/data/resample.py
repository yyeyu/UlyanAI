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

    agg = frame.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        symbol=("symbol", "last"),
        exchange=("exchange", "last"),
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg

