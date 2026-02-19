"""Market data cache for inference service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import get_runtime_paths
from src.features.build import build_features, finalize_features
from src.utils import read_all_parquet

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class CacheEntry:
    data: pd.DataFrame
    fetched_at: datetime
    stale: bool
    source: str


class CandleCache:
    def __init__(self, config: dict[str, Any], root: str | Path = ".") -> None:
        self.config = config
        self.paths = get_runtime_paths(config, root=root)
        self._cache: dict[tuple[str, str], CacheEntry] = {}

    def _asset_info(self, asset: str) -> dict[str, Any]:
        assets = self.config.get("assets", {})
        if asset.upper() not in assets:
            raise ValueError(f"unknown asset: {asset}")
        return assets[asset.upper()]

    def _fetch_binance(self, market_symbol: str, timeframe: str, limit: int = 600) -> pd.DataFrame:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": market_symbol, "interval": timeframe, "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            rows,
            columns=[
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
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = frame[col].astype(float)
        frame["ts_utc"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["close_ts_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        # Only closed candles are allowed by specification.
        frame = frame[frame["close_ts_utc"] <= now].copy()
        return frame[["ts_utc", "open", "high", "low", "close", "volume"]]

    def _load_local_resampled(self, symbol: str, timeframe: str) -> pd.DataFrame:
        folder = (
            self.paths.data_root
            / "resampled"
            / self.config.get("exchange", "binance")
            / symbol.replace("/", "_")
            / timeframe
        )
        data = read_all_parquet(folder)
        if data.empty:
            return data
        return data[["ts_utc", "open", "high", "low", "close", "volume"]]

    def get_candles(self, asset: str, timeframe: str, min_rows: int = 260) -> CacheEntry:
        key = (asset.upper(), timeframe)
        ttl = int(self.config.get("data", {}).get("cache_ttl_seconds", 300))
        now = datetime.now(timezone.utc)
        existing = self._cache.get(key)
        if existing and (now - existing.fetched_at).total_seconds() < ttl and len(existing.data) >= min_rows:
            return existing

        info = self._asset_info(asset)
        try:
            remote = self._fetch_binance(info["market_symbol"], timeframe=timeframe, limit=max(600, min_rows))
            if len(remote) >= min_rows:
                entry = CacheEntry(data=remote, fetched_at=now, stale=False, source="binance")
                self._cache[key] = entry
                return entry
        except Exception:
            pass

        local = self._load_local_resampled(info["symbol"], timeframe=timeframe)
        stale = True
        entry = CacheEntry(data=local, fetched_at=now, stale=stale, source="local_resampled")
        self._cache[key] = entry
        return entry

    def latest_feature_row(self, asset: str, timeframe: str) -> tuple[dict[str, float], float, pd.Timestamp, bool]:
        entry = self.get_candles(asset, timeframe)
        if entry.data.empty:
            raise ValueError(f"no candles available for asset={asset}, timeframe={timeframe}")
        features = finalize_features(
            build_features(entry.data.assign(symbol=asset.upper(), exchange=self.config.get("exchange", "binance")), self.config.get("feature_set_config", {})),
            warmup_drop=True,
        )
        if features.empty:
            raise ValueError(f"not enough candles to build features for asset={asset}, timeframe={timeframe}")
        latest = features.iloc[-1]
        feature_row = {
            col: float(latest[col])
            for col in features.columns
            if col not in {"ts_utc", "symbol", "exchange"}
        }
        spot = float(latest["close"])
        ts = pd.Timestamp(latest["ts_utc"])
        return feature_row, spot, ts, entry.stale

    def latest_price(self, asset: str, timeframe: str = "1m") -> tuple[float, pd.Timestamp, bool]:
        entry = self.get_candles(asset, timeframe=timeframe, min_rows=2)
        if entry.data.empty:
            raise ValueError(f"no candles available for asset={asset}, timeframe={timeframe}")
        latest = entry.data.sort_values("ts_utc").iloc[-1]
        return float(latest["close"]), pd.Timestamp(latest["ts_utc"]), entry.stale

