"""Quality control checks for OHLCV datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import TIMEFRAME_TO_MINUTES


@dataclass(frozen=True)
class QcResult:
    ok: bool
    checks: dict[str, bool]
    details: dict[str, str]


def _check_duplicates(df: pd.DataFrame) -> tuple[bool, str]:
    duplicate_count = int(df["ts_utc"].duplicated().sum())
    return duplicate_count == 0, f"duplicates={duplicate_count}"


def _check_constant_step(df: pd.DataFrame, timeframe: str) -> tuple[bool, str]:
    expected_seconds = TIMEFRAME_TO_MINUTES[timeframe] * 60
    diffs = df["ts_utc"].sort_values().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return True, "single-row dataset"
    bad = int((diffs != expected_seconds).sum())
    return bad == 0, f"bad_steps={bad}, expected_seconds={expected_seconds}"


def _check_missing_ratio(df: pd.DataFrame, timeframe: str) -> tuple[float, int]:
    if len(df) <= 1:
        return 0.0, 0
    expected_seconds = TIMEFRAME_TO_MINUTES[timeframe] * 60
    sorted_ts = df["ts_utc"].sort_values()
    diffs = sorted_ts.diff().dropna().dt.total_seconds()
    gaps = ((diffs / expected_seconds) - 1).clip(lower=0)
    missing = int(gaps.sum())
    ratio = missing / max(len(df) + missing, 1)
    return ratio, missing


def _check_ohlc(df: pd.DataFrame) -> tuple[bool, str]:
    cond = (df["low"] <= df["open"]) & (df["open"] <= df["high"])
    cond &= (df["low"] <= df["close"]) & (df["close"] <= df["high"])
    bad = int((~cond).sum())
    return bad == 0, f"bad_ohlc_rows={bad}"


def _check_volume(df: pd.DataFrame) -> tuple[bool, str]:
    bad = int((df["volume"] < 0).sum())
    return bad == 0, f"negative_volume_rows={bad}"


def _check_timezone(df: pd.DataFrame) -> tuple[bool, str]:
    is_utc = str(df["ts_utc"].dtype).startswith("datetime64[ns, UTC]")
    return is_utc, f"dtype={df['ts_utc'].dtype}"


def _check_no_future(df: pd.DataFrame, now_utc: pd.Timestamp | None = None) -> tuple[bool, str]:
    now = now_utc or pd.Timestamp.now(tz="UTC")
    bad = int((df["ts_utc"] > now).sum())
    return bad == 0, f"future_rows={bad}"


def run_qc(df: pd.DataFrame, *, timeframe: str, max_missing_ratio: float) -> QcResult:
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    checks["duplicates"], details["duplicates"] = _check_duplicates(df)
    checks["constant_step"], details["constant_step"] = _check_constant_step(df, timeframe)
    missing_ratio, missing_rows = _check_missing_ratio(df, timeframe)
    checks["missing_ratio"] = missing_ratio <= max_missing_ratio
    details["missing_ratio"] = f"missing_rows={missing_rows}, ratio={missing_ratio:.6f}"
    checks["ohlc_valid"], details["ohlc_valid"] = _check_ohlc(df)
    checks["volume_valid"], details["volume_valid"] = _check_volume(df)
    checks["timezone_utc"], details["timezone_utc"] = _check_timezone(df)
    checks["no_future"], details["no_future"] = _check_no_future(df)

    ok = all(checks.values())
    return QcResult(ok=ok, checks=checks, details=details)


def assert_qc(df: pd.DataFrame, *, timeframe: str, max_missing_ratio: float) -> None:
    result = run_qc(df, timeframe=timeframe, max_missing_ratio=max_missing_ratio)
    if result.ok:
        return

    failed = [name for name, passed in result.checks.items() if not passed]
    details = "; ".join(f"{name}: {result.details[name]}" for name in failed)
    raise ValueError(f"QC failed for timeframe={timeframe}. failed={failed}. details={details}")
