"""OHLCV download module (Binance Spot) with offline synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils import TIMEFRAME_TO_MINUTES, utc_now

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass(frozen=True)
class DownloadParams:
    symbol_market: str
    interval: str = "1m"
    start: datetime | None = None
    end: datetime | None = None
    limit: int = 1000
    timeout_seconds: int = 30
    offline_fallback: bool = True


def _parse_klines(rows: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    frame[numeric_cols] = frame[numeric_cols].astype(float)
    frame["number_of_trades"] = frame["number_of_trades"].astype(int)
    frame["ts_utc"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_ts_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame["exchange"] = "binance"
    return frame


def _binance_download(params: DownloadParams) -> pd.DataFrame:
    end = params.end or utc_now()
    start = params.start or (end - timedelta(days=30))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows: list[list[Any]] = []
    cursor = start_ms

    step_ms = int(TIMEFRAME_TO_MINUTES.get(params.interval, 1)) * 60_000
    while cursor < end_ms:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": params.symbol_market,
                "interval": params.interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": params.limit,
            },
            timeout=params.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            break
        rows.extend(payload)

        last_open_time = payload[-1][0]
        next_cursor = last_open_time + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(payload) < params.limit:
            break

    frame = _parse_klines(rows)
    if frame.empty:
        raise RuntimeError("Binance returned no klines in requested range")
    return frame


def _synthetic_ohlcv(symbol_market: str, start: datetime, end: datetime) -> pd.DataFrame:
    # Deterministic synthetic fallback for local tests and offline environments.
    ts = pd.date_range(start=start, end=end, freq="1min", tz="UTC", inclusive="left")
    if len(ts) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    base = 50000.0
    returns = rng.normal(0.0, 0.0008, size=len(ts))
    close = base * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.0006, len(ts)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.0006, len(ts)))
    volume = rng.uniform(10.0, 500.0, len(ts))

    frame = pd.DataFrame(
        {
            "open_time": (ts.view("int64") // 10**6).astype(np.int64),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": ((ts + pd.Timedelta(minutes=1)).view("int64") // 10**6).astype(np.int64)
            - 1,
            "quote_asset_volume": volume * close,
            "number_of_trades": rng.integers(100, 3000, len(ts), endpoint=False),
            "taker_buy_base_asset_volume": volume * rng.uniform(0.3, 0.7, len(ts)),
            "taker_buy_quote_asset_volume": volume * close * rng.uniform(0.3, 0.7, len(ts)),
            "ignore": "0",
            "ts_utc": ts,
            "close_ts_utc": ts + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "exchange": "binance",
        }
    )
    frame["symbol_market"] = symbol_market
    return frame


def download_ohlcv(params: DownloadParams) -> pd.DataFrame:
    """Download 1m OHLCV from Binance; fallback to synthetic data if configured."""
    start = params.start or (utc_now() - timedelta(days=30))
    end = params.end or utc_now()
    try:
        frame = _binance_download(params)
        frame["symbol_market"] = params.symbol_market
        return frame
    except Exception:
        if not params.offline_fallback:
            raise
        return _synthetic_ohlcv(
            symbol_market=params.symbol_market,
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
        )
