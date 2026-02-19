from __future__ import annotations

import pandas as pd

from src.data.qc import run_qc
from src.data.resample import resample_ohlcv


def _make_1m(n: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(range(n), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "ts_utc": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
            "symbol": "BTC/USDT",
            "exchange": "binance",
        }
    )


def test_resample_and_qc_5m() -> None:
    candles = _make_1m(60)
    r5 = resample_ohlcv(candles, "5m")
    qc = run_qc(r5, timeframe="5m", max_missing_ratio=0.0)
    assert qc.ok, qc.details
    assert len(r5) == 12


def test_qc_detects_duplicates() -> None:
    candles = _make_1m(10)
    dup = pd.concat([candles, candles.iloc[[0]]], ignore_index=True)
    qc = run_qc(dup, timeframe="1m", max_missing_ratio=0.0)
    assert not qc.ok
    assert not qc.checks["duplicates"]
