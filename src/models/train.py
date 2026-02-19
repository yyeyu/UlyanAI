"""Training pipeline for quantile models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import coverage, mean_interval_width, pinball_loss
from src.labels.build import join_features_labels
from src.models.calibrate import fit_interval_calibrator
from src.models.registry import build_model_version, model_dir, write_metadata
from src.utils import dump_json


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _time_split(df: pd.DataFrame, val_days: int, test_days: int) -> SplitData:
    if df.empty:
        raise ValueError("dataset is empty")
    frame = df.sort_values("ts_utc").reset_index(drop=True)
    max_ts = frame["ts_utc"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = frame[frame["ts_utc"] < val_start]
    val = frame[(frame["ts_utc"] >= val_start) & (frame["ts_utc"] < test_start)]
    test = frame[frame["ts_utc"] >= test_start]

    if len(train) < 200 or len(val) < 50 or len(test) < 50:
        # Fallback ratio split for short local datasets.
        n = len(frame)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        train = frame.iloc[:train_end]
        val = frame.iloc[train_end:val_end]
        test = frame.iloc[val_end:]
    if train.empty or val.empty or test.empty:
        raise ValueError("could not build non-empty train/val/test splits")
    return SplitData(train=train, val=val, test=test)


def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    excluded = {"ts_utc", "symbol", "exchange", label_col}
    cols = [c for c in df.columns if c not in excluded]
    if not cols:
        raise ValueError("no feature columns found")
    return cols


def _train_one_quantile(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    alpha: float,
    train_cfg: dict[str, Any],
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": float(train_cfg.get("learning_rate", 0.05)),
        "num_leaves": int(train_cfg.get("num_leaves", 31)),
        "min_data_in_leaf": int(train_cfg.get("min_data_in_leaf", 20)),
        "seed": int(train_cfg.get("seed", 42)),
        "verbosity": -1,
    }

    ds_train = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
    ds_val = lgb.Dataset(val_df[feature_cols], label=val_df[label_col], reference=ds_train)
    booster = lgb.train(
        params=params,
        train_set=ds_train,
        num_boost_round=int(train_cfg.get("num_boost_round", 120)),
        valid_sets=[ds_val],
        valid_names=["val"],
    )
    return booster


def _predict_triplet(
    models: dict[str, lgb.Booster], frame: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = models["q10"].predict(frame[feature_cols])
    q50 = models["q50"].predict(frame[feature_cols])
    q90 = models["q90"].predict(frame[feature_cols])
    return q10, q50, q90


def _interval_metrics(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    close_price: np.ndarray,
) -> dict[str, float]:
    y_price = close_price * np.exp(y_true)
    low_price = close_price * np.exp(q10)
    high_price = close_price * np.exp(q90)
    return {
        "pinball_q10": pinball_loss(y_true, q10, 0.1),
        "pinball_q50": pinball_loss(y_true, q50, 0.5),
        "pinball_q90": pinball_loss(y_true, q90, 0.9),
        "coverage_return": coverage(y_true, q10, q90),
        "mean_width_return": mean_interval_width(q10, q90),
        "coverage_price": coverage(y_price, low_price, high_price),
        "mean_width_price": mean_interval_width(low_price, high_price),
    }


def train_horizon_model(
    *,
    asset: str,
    horizon: str,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: dict[str, Any],
    artifacts_root: Path,
) -> dict[str, Any]:
    label_col = f"target_r_{horizon}"
    dataset = join_features_labels(features_df, labels_df, label_col)
    dataset = dataset.dropna().reset_index(drop=True)
    if dataset.empty:
        raise ValueError(f"empty dataset for asset={asset}, horizon={horizon}")

    split_cfg = config.get("walk_forward", {})
    splits = _time_split(
        dataset,
        val_days=int(split_cfg.get("val_days", 30)),
        test_days=int(split_cfg.get("test_days", 30)),
    )
    feature_cols = _feature_columns(dataset, label_col)
    train_cfg = {
        **config.get("training", {}),
        "seed": int(config.get("seed", 42)),
    }

    models = {
        "q10": _train_one_quantile(splits.train, splits.val, feature_cols, label_col, 0.1, train_cfg),
        "q50": _train_one_quantile(splits.train, splits.val, feature_cols, label_col, 0.5, train_cfg),
        "q90": _train_one_quantile(splits.train, splits.val, feature_cols, label_col, 0.9, train_cfg),
    }

    y_val = splits.val[label_col].to_numpy()
    close_val = splits.val["close"].to_numpy()
    val_q10, val_q50, val_q90 = _predict_triplet(models, splits.val, feature_cols)
    calibrator = fit_interval_calibrator(
        y_true=y_val,
        q10_pred=val_q10,
        q50_pred=val_q50,
        q90_pred=val_q90,
        target_coverage=0.8,
    )
    val_low_cal, val_high_cal = calibrator.apply(val_q10, val_q50, val_q90)

    y_test = splits.test[label_col].to_numpy()
    close_test = splits.test["close"].to_numpy()
    test_q10, test_q50, test_q90 = _predict_triplet(models, splits.test, feature_cols)
    test_low_cal, test_high_cal = calibrator.apply(test_q10, test_q50, test_q90)

    test_metrics_raw = _interval_metrics(y_test, test_q10, test_q50, test_q90, close_test)
    test_metrics_cal = _interval_metrics(y_test, test_low_cal, test_q50, test_high_cal, close_test)

    naive_q10 = np.quantile(splits.train[label_col].to_numpy(), 0.1)
    naive_q50 = np.quantile(splits.train[label_col].to_numpy(), 0.5)
    naive_q90 = np.quantile(splits.train[label_col].to_numpy(), 0.9)
    naive_metrics = _interval_metrics(
        y_test,
        np.full_like(y_test, naive_q10),
        np.full_like(y_test, naive_q50),
        np.full_like(y_test, naive_q90),
        close_test,
    )

    model_version = build_model_version(
        asset=asset,
        horizon=horizon,
        model="lgbm",
        mode="q",
        feature_set=str(config.get("feature_set", "v1")),
    )
    out_dir = model_dir(artifacts_root, asset, horizon, model_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    models["q10"].save_model(str(out_dir / "model_q10.txt"))
    models["q50"].save_model(str(out_dir / "model_q50.txt"))
    models["q90"].save_model(str(out_dir / "model_q90.txt"))

    dump_json(out_dir / "calibrator.json", calibrator.to_dict())
    metadata = {
        "asset": asset,
        "horizon": horizon,
        "mode": "quantiles_internal_price_ranges_external",
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": config.get("feature_set", "v1"),
        "feature_columns": feature_cols,
        "label_column": label_col,
        "train_rows": int(len(splits.train)),
        "val_rows": int(len(splits.val)),
        "test_rows": int(len(splits.test)),
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
        "naive_baseline_metrics": naive_metrics,
        "calibrator": calibrator.to_dict(),
        "seed": int(config.get("seed", 42)),
    }
    write_metadata(out_dir, metadata)

    return {
        "model_version": model_version,
        "artifact_dir": str(out_dir),
        "metrics_raw": test_metrics_raw,
        "metrics_calibrated": test_metrics_cal,
        "naive_baseline_metrics": naive_metrics,
    }

