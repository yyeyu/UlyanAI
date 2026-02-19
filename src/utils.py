"""Shared utility helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TIMEFRAME_TO_PANDAS_RULE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1W-MON",
}

TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc_index(df: pd.DataFrame, column: str = "ts_utc") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], utc=True)
    out = out.sort_values(column).reset_index(drop=True)
    return out


def yearly_parquet_path(
    data_root: Path,
    layer: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> Path:
    symbol_dir = symbol.replace("/", "_")
    return data_root / layer / exchange / symbol_dir / timeframe / f"{year}.parquet"


def write_partitioned_yearly_parquet(
    df: pd.DataFrame,
    *,
    data_root: Path,
    layer: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    compression: str = "snappy",
) -> list[Path]:
    written: list[Path] = []
    if df.empty:
        return written

    frame = ensure_utc_index(df)
    frame["year"] = frame["ts_utc"].dt.year
    for year, chunk in frame.groupby("year"):
        out_path = yearly_parquet_path(
            data_root=data_root,
            layer=layer,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            year=int(year),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.drop(columns=["year"]).to_parquet(out_path, index=False, compression=compression)
        written.append(out_path)
    return written


def read_all_parquet(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    return ensure_utc_index(combined)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must contain an object at the root")
    return data

