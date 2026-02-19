"""Walk-forward evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import coverage, mean_interval_width, pinball_loss
from src.models.calibrate import fit_interval_calibrator


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    df: pd.DataFrame,
    *,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    max_splits: int,
) -> list[TimeSplit]:
    if df.empty:
        return []
    ts_min = df["ts_utc"].min()
    ts_max = df["ts_utc"].max()
    cursor_train_start = ts_min
    splits: list[TimeSplit] = []

    while len(splits) < max_splits:
        train_end = cursor_train_start + pd.Timedelta(days=train_days)
        val_end = train_end + pd.Timedelta(days=val_days)
        test_end = val_end + pd.Timedelta(days=test_days)
        if test_end > ts_max:
            break
        splits.append(
            TimeSplit(
                train_start=cursor_train_start,
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
            )
        )
        cursor_train_start = cursor_train_start + pd.Timedelta(days=step_days)
    return splits


def _train_quantile(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str], label: str, alpha: float, seed: int) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "seed": seed,
        "verbosity": -1,
    }
    ds_train = lgb.Dataset(train_df[cols], label=train_df[label])
    ds_val = lgb.Dataset(val_df[cols], label=val_df[label], reference=ds_train)
    return lgb.train(params=params, train_set=ds_train, valid_sets=[ds_val], num_boost_round=80)


def run_walk_forward(
    dataset: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    split_cfg: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    frame = dataset.sort_values("ts_utc").reset_index(drop=True)
    splits = build_walk_forward_splits(
        frame,
        train_days=int(split_cfg.get("train_days", 180)),
        val_days=int(split_cfg.get("val_days", 30)),
        test_days=int(split_cfg.get("test_days", 30)),
        step_days=int(split_cfg.get("step_days", 30)),
        max_splits=int(split_cfg.get("max_splits", 3)),
    )
    if not splits:
        # Fallback for short datasets: one ratio-based split.
        n = len(frame)
        if n < 30:
            return {"splits": [], "summary": {}}
        train_end_idx = int(n * 0.7)
        val_end_idx = int(n * 0.85)
        splits = [
            TimeSplit(
                train_start=frame["ts_utc"].iloc[0],
                train_end=frame["ts_utc"].iloc[train_end_idx],
                val_end=frame["ts_utc"].iloc[val_end_idx],
                test_end=frame["ts_utc"].iloc[-1],
            )
        ]

    split_metrics: list[dict[str, Any]] = []
    min_train_rows = 20
    min_val_rows = 5
    min_test_rows = 5
    for idx, split in enumerate(splits, start=1):
        train_df = frame[(frame["ts_utc"] >= split.train_start) & (frame["ts_utc"] < split.train_end)]
        val_df = frame[(frame["ts_utc"] >= split.train_end) & (frame["ts_utc"] < split.val_end)]
        test_df = frame[(frame["ts_utc"] >= split.val_end) & (frame["ts_utc"] < split.test_end)]
        if len(train_df) < min_train_rows or len(val_df) < min_val_rows or len(test_df) < min_test_rows:
            continue

        model_q10 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.1, seed)
        model_q50 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.5, seed)
        model_q90 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.9, seed)

        val_y = val_df[label_col].to_numpy()
        val_q10 = model_q10.predict(val_df[feature_cols])
        val_q50 = model_q50.predict(val_df[feature_cols])
        val_q90 = model_q90.predict(val_df[feature_cols])
        calibrator = fit_interval_calibrator(val_y, val_q10, val_q50, val_q90, target_coverage=0.8)

        test_y = test_df[label_col].to_numpy()
        q10 = model_q10.predict(test_df[feature_cols])
        q50 = model_q50.predict(test_df[feature_cols])
        q90 = model_q90.predict(test_df[feature_cols])
        q10_cal, q90_cal = calibrator.apply(q10, q50, q90)

        split_metrics.append(
            {
                "split_index": idx,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "pinball_q10": pinball_loss(test_y, q10_cal, 0.1),
                "pinball_q50": pinball_loss(test_y, q50, 0.5),
                "pinball_q90": pinball_loss(test_y, q90_cal, 0.9),
                "coverage": coverage(test_y, q10_cal, q90_cal),
                "mean_width": mean_interval_width(q10_cal, q90_cal),
                "calibrator_scale": calibrator.spread_scale,
                "test_period_start": test_df["ts_utc"].min().isoformat(),
                "test_period_end": test_df["ts_utc"].max().isoformat(),
            }
        )

    if not split_metrics:
        # Final fallback: ratio split if configured windows cannot produce usable folds.
        n = len(frame)
        if n < 30:
            return {"splits": [], "summary": {}}
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        train_df = frame.iloc[:train_end]
        val_df = frame.iloc[train_end:val_end]
        test_df = frame.iloc[val_end:]
        if len(train_df) < min_train_rows or len(val_df) < min_val_rows or len(test_df) < min_test_rows:
            return {"splits": [], "summary": {}}

        model_q10 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.1, seed)
        model_q50 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.5, seed)
        model_q90 = _train_quantile(train_df, val_df, feature_cols, label_col, 0.9, seed)

        val_y = val_df[label_col].to_numpy()
        val_q10 = model_q10.predict(val_df[feature_cols])
        val_q50 = model_q50.predict(val_df[feature_cols])
        val_q90 = model_q90.predict(val_df[feature_cols])
        calibrator = fit_interval_calibrator(val_y, val_q10, val_q50, val_q90, target_coverage=0.8)

        test_y = test_df[label_col].to_numpy()
        q10 = model_q10.predict(test_df[feature_cols])
        q50 = model_q50.predict(test_df[feature_cols])
        q90 = model_q90.predict(test_df[feature_cols])
        q10_cal, q90_cal = calibrator.apply(q10, q50, q90)
        split_metrics.append(
            {
                "split_index": 1,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "pinball_q10": pinball_loss(test_y, q10_cal, 0.1),
                "pinball_q50": pinball_loss(test_y, q50, 0.5),
                "pinball_q90": pinball_loss(test_y, q90_cal, 0.9),
                "coverage": coverage(test_y, q10_cal, q90_cal),
                "mean_width": mean_interval_width(q10_cal, q90_cal),
                "calibrator_scale": calibrator.spread_scale,
                "test_period_start": test_df["ts_utc"].min().isoformat(),
                "test_period_end": test_df["ts_utc"].max().isoformat(),
            }
        )

    summary = {
        "splits_count": len(split_metrics),
        "avg_pinball_q10": float(np.mean([m["pinball_q10"] for m in split_metrics])),
        "avg_pinball_q50": float(np.mean([m["pinball_q50"] for m in split_metrics])),
        "avg_pinball_q90": float(np.mean([m["pinball_q90"] for m in split_metrics])),
        "avg_coverage": float(np.mean([m["coverage"] for m in split_metrics])),
        "avg_mean_width": float(np.mean([m["mean_width"] for m in split_metrics])),
    }
    return {"splits": split_metrics, "summary": summary}
