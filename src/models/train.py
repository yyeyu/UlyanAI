"""Training pipeline for quantile models."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import coverage, mean_interval_width, pinball_loss
from src.labels.build import join_features_labels
from src.models.calibrate import fit_interval_calibrator
from src.models.lgbm_compat import sanitize_model_file_in_place
from src.models.registry import build_model_version, model_dir, write_metadata
from src.utils import dump_json


def _quantile_key(alpha: float) -> str:
    pct = int(round(float(alpha) * 100))
    return f"q{pct:02d}"


def _normalize_quantiles(train_cfg: dict[str, Any]) -> list[float]:
    raw = train_cfg.get("quantiles", [0.1, 0.5, 0.9])
    values: set[float] = set()
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            try:
                alpha = float(item)
            except (TypeError, ValueError):
                continue
            if 0.0 < alpha < 1.0:
                values.add(round(alpha, 4))

    # Legacy compatibility: low / mid / high must always exist.
    values.add(0.5)
    for key in ("primary_low_quantile", "primary_high_quantile"):
        try:
            alpha = float(train_cfg.get(key))
        except (TypeError, ValueError):
            continue
        if 0.0 < alpha < 1.0:
            values.add(round(alpha, 4))

    if not values:
        values = {0.1, 0.5, 0.9}
    return sorted(values)


def _primary_quantiles(train_cfg: dict[str, Any], quantiles: list[float]) -> tuple[float, float, float]:
    mid = 0.5
    lower = [alpha for alpha in quantiles if alpha < mid]
    upper = [alpha for alpha in quantiles if alpha > mid]

    try:
        low = float(train_cfg.get("primary_low_quantile", lower[0] if lower else 0.1))
    except (TypeError, ValueError):
        low = lower[0] if lower else 0.1
    try:
        high = float(train_cfg.get("primary_high_quantile", upper[-1] if upper else 0.9))
    except (TypeError, ValueError):
        high = upper[-1] if upper else 0.9

    if low not in quantiles:
        low = max([alpha for alpha in quantiles if alpha < mid], default=min(quantiles))
    if high not in quantiles:
        high = min([alpha for alpha in quantiles if alpha > mid], default=max(quantiles))
    return low, mid, high


def _time_split(
    df: pd.DataFrame,
    val_days: int,
    test_days: int,
    *,
    train_days: int | None = None,
    train_window_mode: str = "expanding",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("dataset is empty")
    frame = df.sort_values("ts_utc").reset_index(drop=True)
    max_ts = frame["ts_utc"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    if str(train_window_mode).lower() == "rolling" and train_days and int(train_days) > 0:
        train_start = val_start - pd.Timedelta(days=int(train_days))
        train = frame[(frame["ts_utc"] >= train_start) & (frame["ts_utc"] < val_start)]
    else:
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
    return train, val, test


def _walk_forward_splits(
    frame: pd.DataFrame,
    *,
    train_days: int,
    val_days: int,
    step_days: int,
    max_folds: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if frame.empty:
        return []
    train_days_clean = max(1, int(train_days))
    val_days_clean = max(1, int(val_days))
    step_days_clean = max(1, int(step_days))
    max_folds_clean = max(1, int(max_folds))

    train_delta = pd.Timedelta(days=train_days_clean)
    val_delta = pd.Timedelta(days=val_days_clean)
    step_delta = pd.Timedelta(days=step_days_clean)

    ts_min = pd.Timestamp(frame["ts_utc"].min())
    ts_max = pd.Timestamp(frame["ts_utc"].max())
    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = ts_min
    for _ in range(max_folds_clean):
        train_end = train_start + train_delta
        val_start = train_end
        val_end = val_start + val_delta
        if val_end > ts_max:
            break
        splits.append((train_start, train_end, val_start, val_end))
        train_start = train_start + step_delta
    return splits


def evaluate_walk_forward_horizon(
    *,
    asset: str,
    horizon: str,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: dict[str, Any],
    progress_hook: Callable[[str, float], None] | None = None,
    abort_if: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    label_col = f"target_r_{horizon}"
    dataset = join_features_labels(features_df, labels_df, label_col)
    dataset = dataset.dropna().sort_values("ts_utc").reset_index(drop=True)
    if dataset.empty:
        raise ValueError(f"empty dataset for walk-forward asset={asset}, horizon={horizon}")

    train_cfg = {
        **config.get("training", {}),
        "seed": int(config.get("seed", 42)),
    }
    wf_cfg = dict(config.get("walk_forward", {}))
    train_days = int(wf_cfg.get("train_days", 180) or 180)
    val_days = int(wf_cfg.get("val_days", 30) or 30)
    step_days = int(wf_cfg.get("step_days", 30) or 30)
    max_folds = int(wf_cfg.get("max_splits", 3) or 3)
    target_coverage = float(train_cfg.get("target_coverage", 0.8))
    quantiles = _normalize_quantiles(train_cfg)
    low_alpha, mid_alpha, high_alpha = _primary_quantiles(train_cfg, quantiles)
    low_key = _quantile_key(low_alpha)
    mid_key = _quantile_key(mid_alpha)
    high_key = _quantile_key(high_alpha)
    feature_cols = _feature_columns(dataset, label_col)

    splits = _walk_forward_splits(
        dataset,
        train_days=train_days,
        val_days=val_days,
        step_days=step_days,
        max_folds=max_folds,
    )
    if not splits:
        raise ValueError(
            "walk-forward could not build folds; adjust wf_train_days/wf_val_days/wf_step_days/wf_folds"
        )

    min_train_rows = max(50, len(feature_cols) + 5)
    min_val_rows = max(20, len(feature_cols) + 5)
    folds: list[dict[str, Any]] = []
    total_splits = max(1, len(splits))

    for fold_index, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        if abort_if and abort_if():
            raise RuntimeError("training cancelled")

        train_df = dataset[(dataset["ts_utc"] >= train_start) & (dataset["ts_utc"] < train_end)]
        val_df = dataset[(dataset["ts_utc"] >= val_start) & (dataset["ts_utc"] < val_end)]
        if len(train_df) < min_train_rows or len(val_df) < min_val_rows:
            continue

        train_last_ts = pd.Timestamp(train_df["ts_utc"].max())
        val_first_ts = pd.Timestamp(val_df["ts_utc"].min())
        if train_last_ts >= val_first_ts:
            raise ValueError(
                f"walk-forward leakage detected for horizon={horizon}: train_max_ts >= val_min_ts"
            )

        models: dict[str, lgb.Booster] = {}
        for alpha in quantiles:
            if abort_if and abort_if():
                raise RuntimeError("training cancelled")
            key = _quantile_key(alpha)
            models[key] = _train_one_quantile(train_df, val_df, feature_cols, label_col, alpha, train_cfg)

        val_predictions = _predict_quantiles(models, val_df, feature_cols)
        val_y = val_df[label_col].to_numpy()
        close_val = val_df["close"].to_numpy()
        metrics_raw = _interval_metrics(
            val_y,
            val_predictions[low_key],
            val_predictions[mid_key],
            val_predictions[high_key],
            close_val,
            low_alpha=low_alpha,
            high_alpha=high_alpha,
        )
        calibrator = fit_interval_calibrator(
            y_true=val_y,
            q10_pred=val_predictions[low_key],
            q50_pred=val_predictions[mid_key],
            q90_pred=val_predictions[high_key],
            target_coverage=target_coverage,
            candidate_scales=train_cfg.get("scale_grid"),
            selection_rule=str(train_cfg.get("scale_selection_rule", "min_abs_coverage_gap_then_width")),
            score_weight_gap=float(train_cfg.get("scale_score_weight_gap", 1.0) or 1.0),
            score_weight_width=float(train_cfg.get("scale_score_weight_width", 1.0) or 1.0),
        )
        val_low_cal, val_high_cal = calibrator.apply(
            val_predictions[low_key],
            val_predictions[mid_key],
            val_predictions[high_key],
        )
        metrics_calibrated = _interval_metrics(
            val_y,
            val_low_cal,
            val_predictions[mid_key],
            val_high_cal,
            close_val,
            low_alpha=low_alpha,
            high_alpha=high_alpha,
        )

        folds.append(
            {
                "fold_index": fold_index,
                "train_start": _ts_iso(train_start),
                "train_end": _ts_iso(train_end),
                "val_start": _ts_iso(val_start),
                "val_end": _ts_iso(val_end),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "train_max_ts": _ts_iso(train_last_ts),
                "val_min_ts": _ts_iso(val_first_ts),
                "metrics_raw": metrics_raw,
                "metrics_calibrated": metrics_calibrated,
                "calibrator": {
                    "selected_scale": float(calibrator.spread_scale),
                    "chosen_scale": float(calibrator.chosen_scale),
                    "target_coverage": float(calibrator.target_coverage),
                },
            }
        )

        if progress_hook:
            progress_hook("walk_forward", fold_index / total_splits)

    if not folds:
        raise ValueError(
            "walk-forward produced no usable folds; increase data or reduce wf_train_days/wf_val_days"
        )

    avg_raw: dict[str, float] = {}
    avg_calibrated: dict[str, float] = {}
    raw_keys = sorted({key for fold in folds for key in fold["metrics_raw"].keys()})
    cal_keys = sorted({key for fold in folds for key in fold["metrics_calibrated"].keys()})
    for key in raw_keys:
        values = [float(fold["metrics_raw"][key]) for fold in folds if key in fold["metrics_raw"]]
        if values:
            avg_raw[key] = float(np.mean(values))
    for key in cal_keys:
        values = [float(fold["metrics_calibrated"][key]) for fold in folds if key in fold["metrics_calibrated"]]
        if values:
            avg_calibrated[key] = float(np.mean(values))

    return {
        "asset": asset.upper(),
        "horizon": horizon,
        "label_column": label_col,
        "quantiles": [_quantile_key(alpha) for alpha in quantiles],
        "primary_interval": {
            "low": low_key,
            "mid": mid_key,
            "high": high_key,
        },
        "config": {
            "train_days": train_days,
            "val_days": val_days,
            "step_days": step_days,
            "max_folds": max_folds,
            "target_coverage": target_coverage,
            "train_window_mode": "rolling",
        },
        "folds": folds,
        "summary": {
            "folds_requested": int(max_folds),
            "folds_built": int(len(splits)),
            "folds_evaluated": int(len(folds)),
            "avg_raw": avg_raw,
            "avg_calibrated": avg_calibrated,
        },
    }


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
    params: dict[str, Any] = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": float(train_cfg.get("learning_rate", 0.05)),
        "num_leaves": int(train_cfg.get("num_leaves", 31)),
        "min_data_in_leaf": int(train_cfg.get("min_data_in_leaf", 20)),
        "seed": int(train_cfg.get("seed", 42)),
        "verbosity": -1,
    }

    optional_params = {
        "max_depth": int,
        "feature_fraction": float,
        "bagging_fraction": float,
        "bagging_freq": int,
        "lambda_l1": float,
        "lambda_l2": float,
    }
    for key, caster in optional_params.items():
        value = train_cfg.get(key)
        if value in (None, ""):
            continue
        params[key] = caster(value)

    callbacks: list[Callable[..., Any]] = []
    try:
        early_stopping_rounds = int(train_cfg.get("early_stopping_rounds", 0) or 0)
    except (TypeError, ValueError):
        early_stopping_rounds = 0
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    ds_train = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
    ds_val = lgb.Dataset(val_df[feature_cols], label=val_df[label_col], reference=ds_train)
    return lgb.train(
        params=params,
        train_set=ds_train,
        num_boost_round=int(train_cfg.get("num_boost_round", train_cfg.get("n_estimators", 120))),
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=callbacks,
    )


def _predict_quantiles(
    models: dict[str, lgb.Booster],
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    return {
        key: model.predict(frame[feature_cols])
        for key, model in models.items()
    }


def _interval_metrics(
    y_true: np.ndarray,
    low_pred: np.ndarray,
    mid_pred: np.ndarray,
    high_pred: np.ndarray,
    close_price: np.ndarray,
    *,
    low_alpha: float,
    high_alpha: float,
) -> dict[str, float]:
    y_price = close_price * np.exp(y_true)
    low_price = close_price * np.exp(low_pred)
    high_price = close_price * np.exp(high_pred)
    return {
        f"pinball_{_quantile_key(low_alpha)}": pinball_loss(y_true, low_pred, low_alpha),
        "pinball_q50": pinball_loss(y_true, mid_pred, 0.5),
        f"pinball_{_quantile_key(high_alpha)}": pinball_loss(y_true, high_pred, high_alpha),
        "coverage_return": coverage(y_true, low_pred, high_pred),
        "mean_width_return": mean_interval_width(low_pred, high_pred),
        "coverage_price": coverage(y_price, low_price, high_price),
        "mean_width_price": mean_interval_width(low_price, high_price),
    }


def _ts_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    return text or None


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def train_horizon_model(
    *,
    asset: str,
    horizon: str,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: dict[str, Any],
    artifacts_root: Path,
    progress_hook: Callable[[str, float], None] | None = None,
    abort_if: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    label_col = f"target_r_{horizon}"
    dataset = join_features_labels(features_df, labels_df, label_col)
    dataset = dataset.dropna().reset_index(drop=True)
    if dataset.empty:
        raise ValueError(f"empty dataset for asset={asset}, horizon={horizon}")

    split_cfg = config.get("walk_forward", {})
    train_cfg = {
        **config.get("training", {}),
        "seed": int(config.get("seed", 42)),
    }
    quantiles = _normalize_quantiles(train_cfg)
    low_alpha, mid_alpha, high_alpha = _primary_quantiles(train_cfg, quantiles)
    train_df, val_df, test_df = _time_split(
        dataset,
        val_days=int(train_cfg.get("val_days", split_cfg.get("val_days", 30))),
        test_days=int(train_cfg.get("test_days", split_cfg.get("test_days", 30))),
        train_days=int(train_cfg.get("train_window_days", 0) or 0),
        train_window_mode=str(train_cfg.get("train_window_mode", "expanding")),
    )
    feature_cols = _feature_columns(dataset, label_col)

    models: dict[str, lgb.Booster] = {}
    total_quantiles = max(1, len(quantiles))
    for index, alpha in enumerate(quantiles, start=1):
        if abort_if and abort_if():
            raise RuntimeError("training cancelled")
        key = _quantile_key(alpha)
        if progress_hook:
            progress_hook(f"training_{key}", index / total_quantiles)
        models[key] = _train_one_quantile(train_df, val_df, feature_cols, label_col, alpha, train_cfg)

    val_predictions = _predict_quantiles(models, val_df, feature_cols)
    low_key = _quantile_key(low_alpha)
    high_key = _quantile_key(high_alpha)
    mid_key = _quantile_key(mid_alpha)
    builder_metadata = {
        key: value
        for key, value in dict(config.get("builder_metadata", {})).items()
        if value not in (None, "", [], {})
    }
    walk_forward_enabled = bool(builder_metadata.get("walk_forward"))
    selection_rule = str(
        builder_metadata.get(
            "scale_selection_rule",
            train_cfg.get("scale_selection_rule", "min_abs_coverage_gap_then_width"),
        )
    )
    score_weight_gap = _to_float(
        builder_metadata.get("scale_score_weight_gap", train_cfg.get("scale_score_weight_gap", 1.0)),
        1.0,
    )
    score_weight_width = _to_float(
        builder_metadata.get("scale_score_weight_width", train_cfg.get("scale_score_weight_width", 1.0)),
        1.0,
    )
    calibrator = fit_interval_calibrator(
        y_true=val_df[label_col].to_numpy(),
        q10_pred=val_predictions[low_key],
        q50_pred=val_predictions[mid_key],
        q90_pred=val_predictions[high_key],
        target_coverage=float(train_cfg.get("target_coverage", 0.8)),
        candidate_scales=train_cfg.get("scale_grid"),
        selection_rule=selection_rule,
        score_weight_gap=score_weight_gap,
        score_weight_width=score_weight_width,
    )

    train_predictions = _predict_quantiles(models, train_df, feature_cols)
    train_metrics_raw = _interval_metrics(
        train_df[label_col].to_numpy(),
        train_predictions[low_key],
        train_predictions[mid_key],
        train_predictions[high_key],
        train_df["close"].to_numpy(),
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )

    val_low_cal, val_high_cal = calibrator.apply(
        val_predictions[low_key],
        val_predictions[mid_key],
        val_predictions[high_key],
    )
    val_metrics_raw = _interval_metrics(
        val_df[label_col].to_numpy(),
        val_predictions[low_key],
        val_predictions[mid_key],
        val_predictions[high_key],
        val_df["close"].to_numpy(),
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )
    val_metrics_cal = _interval_metrics(
        val_df[label_col].to_numpy(),
        val_low_cal,
        val_predictions[mid_key],
        val_high_cal,
        val_df["close"].to_numpy(),
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )

    test_predictions = _predict_quantiles(models, test_df, feature_cols)
    test_low_cal, test_high_cal = calibrator.apply(
        test_predictions[low_key],
        test_predictions[mid_key],
        test_predictions[high_key],
    )

    y_test = test_df[label_col].to_numpy()
    close_test = test_df["close"].to_numpy()
    test_metrics_raw = _interval_metrics(
        y_test,
        test_predictions[low_key],
        test_predictions[mid_key],
        test_predictions[high_key],
        close_test,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )
    test_metrics_cal = _interval_metrics(
        y_test,
        test_low_cal,
        test_predictions[mid_key],
        test_high_cal,
        close_test,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )

    train_values = train_df[label_col].to_numpy()
    naive_low = np.quantile(train_values, low_alpha)
    naive_mid = np.quantile(train_values, 0.5)
    naive_high = np.quantile(train_values, high_alpha)
    naive_metrics = _interval_metrics(
        y_test,
        np.full_like(y_test, naive_low),
        np.full_like(y_test, naive_mid),
        np.full_like(y_test, naive_high),
        close_test,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
    )

    model_version = build_model_version(
        asset=asset,
        horizon=horizon,
        model="lgbm",
        mode="q",
        feature_set=str(config.get("feature_set", "v1")),
    )
    out_dir = model_dir(artifacts_root, asset, horizon, model_version)
    if out_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%H%M%S")
        model_version = f"{model_version}_{suffix}"
        out_dir = model_dir(artifacts_root, asset, horizon, model_version)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, booster in models.items():
        model_path = out_dir / f"model_{key}.txt"
        booster.save_model(str(model_path))
        sanitize_model_file_in_place(model_path)

    dump_json(out_dir / "calibrator.json", calibrator.to_dict())
    wf_report: dict[str, Any] | None = None
    if walk_forward_enabled:
        if progress_hook:
            progress_hook("walk_forward_eval", 0.0)
        wf_report = evaluate_walk_forward_horizon(
            asset=asset,
            horizon=horizon,
            features_df=features_df,
            labels_df=labels_df,
            config=config,
            progress_hook=progress_hook,
            abort_if=abort_if,
        )
        dump_json(out_dir / "wf_report.json", wf_report)
        if progress_hook:
            progress_hook("walk_forward_eval", 1.0)
    base_mode = str(builder_metadata.get("base_timeframe_mode", "legacy")).strip().lower()
    base_timeframe = (
        "1m"
        if base_mode == "base_1m"
        else str(config.get("horizons_map", {}).get(horizon, {}).get("timeframe", horizon))
    )
    steps_runtime = dict(builder_metadata.get("steps_runtime", {}))
    try:
        steps_ahead = int(steps_runtime.get(horizon, 1))
    except (TypeError, ValueError):
        steps_ahead = 1
    params_json = {
        "asset": asset.upper(),
        "horizon": horizon,
        "base_timeframe": base_timeframe,
        "steps_ahead": steps_ahead,
        "quantiles": [_quantile_key(alpha) for alpha in quantiles],
        "target_coverage_requested": float(
            builder_metadata.get("requested_coverage", train_cfg.get("target_coverage", 0.8))
        ),
        "target_coverage_effective": float(
            builder_metadata.get("effective_coverage", train_cfg.get("target_coverage", 0.8))
        ),
        "interval_mode": str(builder_metadata.get("interval_mode", "symmetric")),
        "train_window_mode": str(train_cfg.get("train_window_mode", "expanding")),
        "train_window_days": int(train_cfg.get("train_window_days", 365) or 365),
        "val_days": int(train_cfg.get("val_days", split_cfg.get("val_days", 30))),
        "test_days": int(train_cfg.get("test_days", split_cfg.get("test_days", 30))),
        "walk_forward": (
            {
                **dict(builder_metadata.get("walk_forward", {})),
                "report_path": "wf_report.json",
                "summary": dict(wf_report.get("summary", {})) if wf_report else {},
            }
            if walk_forward_enabled
            else None
        ),
        "feature_set_version": str(builder_metadata.get("feature_set_label", config.get("feature_set", "v1"))),
        "calibration": {
            "method": str(builder_metadata.get("calibration_method", "grid_scale")),
            "scale_grid": list(train_cfg.get("scale_grid", []) or []),
            "selection_rule": str(calibrator.selection_rule),
            "selected_scale": float(calibrator.spread_scale),
            "chosen_scale": float(calibrator.chosen_scale),
            "score_weight_gap": float(calibrator.score_weight_gap),
            "score_weight_width": float(calibrator.score_weight_width),
            "scale_table": [dict(item) for item in calibrator.scale_table],
            "target_coverage": float(calibrator.target_coverage),
        },
        "seed": int(config.get("seed", 42)),
        "data_range_start": _ts_iso(dataset["ts_utc"].min()),
        "data_range_end": _ts_iso(dataset["ts_utc"].max()),
    }
    registry_metrics = {
        "train": train_metrics_raw,
        "val": val_metrics_raw,
        "test": test_metrics_raw,
        "calibrated": test_metrics_cal,
        "val_calibrated": val_metrics_cal,
    }
    created_at_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "asset": asset,
        "horizon": horizon,
        "mode": "quantiles_internal_price_ranges_external",
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "feature_set": config.get("feature_set", "v1"),
        "feature_columns": feature_cols,
        "label_column": label_col,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "trained_quantiles": [_quantile_key(alpha) for alpha in quantiles],
        "primary_low_quantile": low_alpha,
        "primary_mid_quantile": mid_alpha,
        "primary_high_quantile": high_alpha,
        "train_metrics_raw": train_metrics_raw,
        "val_metrics_raw": val_metrics_raw,
        "val_metrics_calibrated": val_metrics_cal,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
        "registry_metrics": registry_metrics,
        "params_json": params_json,
        "base_timeframe": base_timeframe,
        "steps_ahead": steps_ahead,
        "train_window_mode": str(train_cfg.get("train_window_mode", "expanding")),
        "train_window_days": int(train_cfg.get("train_window_days", 365) or 365),
        "val_days": int(train_cfg.get("val_days", split_cfg.get("val_days", 30))),
        "test_days": int(train_cfg.get("test_days", split_cfg.get("test_days", 30))),
        "data_range_start": _ts_iso(dataset["ts_utc"].min()),
        "data_range_end": _ts_iso(dataset["ts_utc"].max()),
        "naive_baseline_metrics": naive_metrics,
        "calibrator": calibrator.to_dict(),
        "walk_forward_report_path": "wf_report.json" if wf_report else None,
        "walk_forward_report_summary": dict(wf_report.get("summary", {})) if wf_report else {},
        "seed": int(config.get("seed", 42)),
    }
    metadata.update(builder_metadata)
    write_metadata(out_dir, metadata)

    return {
        "model_version": model_version,
        "artifact_dir": str(out_dir),
        "created_at_utc": created_at_utc,
        "trained_quantiles": metadata["trained_quantiles"],
        "primary_interval": {
            "low": _quantile_key(low_alpha),
            "mid": _quantile_key(mid_alpha),
            "high": _quantile_key(high_alpha),
        },
        "registry_metrics": registry_metrics,
        "params_json": params_json,
        "metrics_raw": test_metrics_raw,
        "metrics_calibrated": test_metrics_cal,
        "naive_baseline_metrics": naive_metrics,
        "wf_report_path": str(out_dir / "wf_report.json") if wf_report else None,
        "wf_report_summary": dict(wf_report.get("summary", {})) if wf_report else {},
    }
