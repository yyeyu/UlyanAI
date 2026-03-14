"""Model Builder validation and background training helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_runtime_paths
from src.features.build import build_features, finalize_features
from src.labels.build import build_labels
from src.models.train import evaluate_walk_forward_horizon, train_horizon_model
from src.pipeline import run_data_pipeline, run_feature_label_pipeline
from src.utils import dump_json, read_all_parquet

SUPPORTED_MVP_HORIZONS = ("5m", "15m", "1h")
AUTO_STEPS_AHEAD = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}
SOFT_QUANTILE_LIMIT = 11
DEFAULT_SCALE_GRID = [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
SWEEP_VARIANT_CONFIRM_LIMIT = 50
SWEEP_VARIANT_HARD_LIMIT = 200
HYPERPARAM_SPECS: dict[str, dict[str, Any]] = {
    "learning_rate": {"kind": "float", "minimum": 0.0001, "maximum": 10.0},
    "num_leaves": {"kind": "int", "minimum": 2, "maximum": 131072},
    "max_depth": {"kind": "int", "minimum": -1, "maximum": 4096},
    "min_data_in_leaf": {"kind": "int", "minimum": 1, "maximum": 1000000},
    "feature_fraction": {"kind": "float", "minimum": 0.0, "maximum": 1.0},
    "bagging_fraction": {"kind": "float", "minimum": 0.0, "maximum": 1.0},
    "bagging_freq": {"kind": "int", "minimum": 0, "maximum": 100000},
    "lambda_l1": {"kind": "float", "minimum": 0.0, "maximum": 1000000.0},
    "lambda_l2": {"kind": "float", "minimum": 0.0, "maximum": 1000000.0},
    "num_boost_round": {"kind": "int", "minimum": 1, "maximum": 1000000},
    "early_stopping_rounds": {"kind": "int", "minimum": 0, "maximum": 1000000},
    "seed": {"kind": "int", "minimum": -2147483648, "maximum": 2147483647},
}


class TrainingJobCancelled(RuntimeError):
    """Raised when a running training job is cancelled."""


def _normalize_training_budget_preset(raw: Any) -> str:
    value = str(raw or "custom").strip()
    value_upper = value.upper()
    if value_upper in {"F0", "F1", "F2"}:
        return value_upper
    return "custom"


def _quantile_key_from_percent(percent: int) -> str:
    return f"q{int(percent):02d}"


def _quantile_percent_from_key(key: str) -> int | None:
    raw = str(key).strip().lower()
    if not raw.startswith("q") or len(raw) != 3:
        return None
    try:
        value = int(raw[1:])
    except ValueError:
        return None
    if 1 <= value <= 99:
        return value
    return None


def _sorted_quantile_keys(keys: set[str]) -> list[str]:
    return sorted(keys, key=lambda item: int(item[1:]))


def _dedupe_preserve(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _coerce_hyperparam_value(name: str, raw: Any) -> float | int | None:
    spec = HYPERPARAM_SPECS.get(str(name))
    if not spec:
        return None
    kind = str(spec["kind"])
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    try:
        numeric = int(raw) if kind == "int" else float(raw)
    except (TypeError, ValueError):
        return None
    if kind == "int":
        if not isinstance(numeric, int):
            return None
    if minimum is not None and numeric < minimum:
        return None
    if maximum is not None and numeric > maximum:
        return None
    if kind == "float":
        return round(float(numeric), 10)
    return int(numeric)


def _normalize_hyperparams(
    raw: Any,
    *,
    warnings: list[str] | None = None,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if name not in HYPERPARAM_SPECS:
            if warnings is not None:
                warnings.append(f"hyperparam {name} is not supported and was ignored")
            continue
        cleaned = _coerce_hyperparam_value(name, value)
        if cleaned is None:
            if warnings is not None:
                warnings.append(f"hyperparam {name} had an invalid value and was ignored")
            continue
        out[name] = cleaned
    return out


def _normalize_sweep_hyperparams(
    raw: Any,
    *,
    warnings: list[str] | None = None,
) -> dict[str, list[float | int]]:
    out: dict[str, list[float | int]] = {}
    if not isinstance(raw, dict):
        return out
    for key, values in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if name not in HYPERPARAM_SPECS:
            if warnings is not None:
                warnings.append(f"sweep hyperparam {name} is not supported and was ignored")
            continue
        if not isinstance(values, (list, tuple, set)):
            continue
        seen: set[float | int] = set()
        normalized_values: list[float | int] = []
        for item in values:
            cleaned = _coerce_hyperparam_value(name, item)
            if cleaned is None or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized_values.append(cleaned)
        if normalized_values:
            out[name] = normalized_values
    return out


def _normalized_sweep_dimensions(
    payload: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    interval_mode = str(payload.get("interval_mode", "symmetric")).strip().lower()

    coverages = _dedupe_preserve(
        _coerce_int_list(payload.get("sweep_target_coverages"), minimum=1, maximum=99)
    )
    if interval_mode == "custom":
        coverages = []
    if not coverages:
        try:
            coverages = [max(1, min(99, int(payload.get("target_coverage_percent", 80) or 80)))]
        except (TypeError, ValueError):
            coverages = [80]

    train_window_days = _dedupe_preserve(
        _coerce_int_list(payload.get("sweep_train_window_days"), minimum=1, maximum=3650)
    )
    if not train_window_days:
        try:
            train_window_days = [max(1, min(3650, int(payload.get("train_window_days", 365) or 365)))]
        except (TypeError, ValueError):
            train_window_days = [365]

    feature_set_versions = _dedupe_preserve(
        [
            str(item).strip()
            for item in payload.get("sweep_feature_set_versions", [])
            if str(item).strip()
        ]
    )
    if not feature_set_versions:
        feature_set_versions = [str(payload.get("feature_set_version", "feat_v1")).strip() or "feat_v1"]

    calibration_methods = _dedupe_preserve(
        [
            str(item).strip().lower()
            for item in payload.get("sweep_calibration_methods", [])
            if str(item).strip()
        ]
    )
    if not calibration_methods:
        calibration_methods = [str(payload.get("calibration_method", "grid_scale")).strip().lower() or "grid_scale"]

    sweep_hyperparams = _normalize_sweep_hyperparams(payload.get("sweep_hyperparams", {}), warnings=warnings)

    return {
        "target_coverages": coverages,
        "train_window_days": train_window_days,
        "feature_set_versions": feature_set_versions,
        "calibration_methods": calibration_methods,
        "hyperparams": sweep_hyperparams,
    }


def _estimate_sweep_variant_count_from_dimensions(
    payload: dict[str, Any],
    dimensions: dict[str, Any],
) -> int:
    interval_mode = str(payload.get("interval_mode", "symmetric")).strip().lower()

    if interval_mode == "custom":
        coverage_count = 1
    else:
        coverage_signatures: set[tuple[int, int]] = set()
        for coverage_percent in dimensions["target_coverages"]:
            q_low, q_high, _ = _effective_symmetric_coverage(int(coverage_percent))
            coverage_signatures.add((q_low, q_high))
        coverage_count = max(1, len(coverage_signatures))

    feature_sets = {
        _normalize_feature_set_version(item)[0]
        for item in dimensions["feature_set_versions"]
    }
    calibration_methods = {
        "grid_scale" if str(item).strip().lower() != "grid_scale" else "grid_scale"
        for item in dimensions["calibration_methods"]
    }

    total = coverage_count
    total *= max(1, len(dimensions["train_window_days"]))
    total *= max(1, len(feature_sets))
    total *= max(1, len(calibration_methods))
    for values in dict(dimensions.get("hyperparams", {})).values():
        total *= max(1, len(values))
    return max(1, total)


def estimate_sweep_variant_count(payload: dict[str, Any]) -> int:
    dimensions = _normalized_sweep_dimensions(payload)
    return _estimate_sweep_variant_count_from_dimensions(payload, dimensions)


def _normalize_feature_set_version(raw: Any) -> tuple[str, str | None]:
    text = str(raw or "feat_v1").strip().lower()
    mapping = {
        "feat_v1": "v1",
        "v1": "v1",
    }
    if text in mapping:
        return mapping[text], None
    return "v1", f"feature_set_version={text} not found, falling back to feat_v1"


def _normalize_tags(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {
            str(key).strip(): str(value).strip()
            for key, value in raw.items()
            if str(key).strip()
        }
    out: dict[str, str] = {}
    if isinstance(raw, str):
        rows = [item.strip() for item in raw.replace(",", "\n").splitlines() if item.strip()]
        for row in rows:
            if ":" not in row:
                continue
            key, value = row.split(":", 1)
            key_clean = key.strip()
            if not key_clean:
                continue
            out[key_clean] = value.strip()
    return out


def _coerce_int_list(raw: Any, *, minimum: int = 1, maximum: int = 10000) -> list[int]:
    out: list[int] = []
    if not isinstance(raw, (list, tuple, set)):
        return out
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if minimum <= value <= maximum:
            out.append(value)
    return out


def _effective_symmetric_coverage(target_coverage_percent: int) -> tuple[int, int, float]:
    q_low = int(round((100 - int(target_coverage_percent)) / 2))
    q_low = max(1, min(49, q_low))
    q_high = 100 - q_low
    effective = (q_high - q_low) / 100.0
    return q_low, q_high, effective


def _build_feature_config(base_feature_cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    feature_cfg = deepcopy(base_feature_cfg)
    groups = {
        "returns": True,
        "volatility": True,
        "volume": True,
        "trend": True,
        "rsi": True,
        "atr": True,
        "macd": True,
    }
    groups.update(
        {
            str(key): bool(value)
            for key, value in dict(payload.get("feature_groups", {})).items()
        }
    )
    overrides = dict(payload.get("feature_overrides", {}))

    if not groups.get("returns", True):
        feature_cfg["return_windows"] = []
    elif "return_windows" in overrides:
        feature_cfg["return_windows"] = _coerce_int_list(overrides.get("return_windows"), minimum=1)

    if not groups.get("volatility", True):
        feature_cfg["vol_windows"] = []
    elif "vol_windows" in overrides:
        feature_cfg["vol_windows"] = _coerce_int_list(overrides.get("vol_windows"), minimum=2)

    if not groups.get("volume", True):
        feature_cfg["volume_windows"] = []
    elif "volume_windows" in overrides:
        feature_cfg["volume_windows"] = _coerce_int_list(overrides.get("volume_windows"), minimum=2)

    if not groups.get("trend", True):
        feature_cfg["trend_windows"] = []
    elif "trend_windows" in overrides:
        feature_cfg["trend_windows"] = _coerce_int_list(overrides.get("trend_windows"), minimum=2)

    if not groups.get("rsi", True):
        feature_cfg["rsi_window"] = 2
    elif "rsi_window" in overrides:
        values = _coerce_int_list([overrides.get("rsi_window")], minimum=2)
        if values:
            feature_cfg["rsi_window"] = values[0]

    if not groups.get("atr", True):
        feature_cfg["atr_window"] = 2
    elif "atr_window" in overrides:
        values = _coerce_int_list([overrides.get("atr_window")], minimum=2)
        if values:
            feature_cfg["atr_window"] = values[0]

    if not groups.get("macd", True):
        feature_cfg["macd"] = {"fast": 3, "slow": 3, "signal": 2}
    elif "macd" in overrides and isinstance(overrides.get("macd"), dict):
        raw_macd = dict(overrides["macd"])
        fast = _coerce_int_list([raw_macd.get("fast")], minimum=2)
        slow = _coerce_int_list([raw_macd.get("slow")], minimum=2)
        signal = _coerce_int_list([raw_macd.get("signal")], minimum=2)
        feature_cfg["macd"] = {
            "fast": fast[0] if fast else int(feature_cfg.get("macd", {}).get("fast", 12)),
            "slow": slow[0] if slow else int(feature_cfg.get("macd", {}).get("slow", 26)),
            "signal": signal[0] if signal else int(feature_cfg.get("macd", {}).get("signal", 9)),
        }

    return feature_cfg


@dataclass(frozen=True)
class BuilderValidation:
    response: dict[str, Any]
    normalized: dict[str, Any]


def validate_builder_payload(
    payload: dict[str, Any],
    *,
    require_confirmation: bool = False,
    job_type: str | None = None,
) -> BuilderValidation:
    warnings: list[str] = []
    errors: list[str] = []

    asset = str(payload.get("asset", "BTC")).upper()
    if asset != "BTC":
        warnings.append("MVP currently targets BTC; using requested asset if data exists")

    raw_training_budget_preset = payload.get("training_budget_preset", "custom")
    raw_training_budget_text = str(raw_training_budget_preset or "").strip()
    training_budget_preset = _normalize_training_budget_preset(raw_training_budget_preset)
    if raw_training_budget_text and raw_training_budget_text.upper() not in {"CUSTOM", "F0", "F1", "F2"}:
        warnings.append("training_budget_preset reset to custom")
        training_budget_preset = "custom"

    requested_horizons = [str(item) for item in payload.get("horizons", []) if item]
    if not requested_horizons:
        requested_horizons = ["5m"]
    seen_horizons: set[str] = set()
    horizons: list[str] = []
    for item in requested_horizons:
        if item in seen_horizons:
            continue
        seen_horizons.add(item)
        horizons.append(item)
        if item not in SUPPORTED_MVP_HORIZONS:
            warnings.append(f"horizon={item} is outside current MVP focus")

    base_timeframe_mode = str(payload.get("base_timeframe_mode", "legacy")).strip().lower()
    if base_timeframe_mode not in {"legacy", "base_1m"}:
        base_timeframe_mode = "legacy"
        warnings.append("base_timeframe_mode reset to legacy")

    try:
        target_coverage_percent = int(payload.get("target_coverage_percent", 80))
    except (TypeError, ValueError):
        target_coverage_percent = 80
        warnings.append("target_coverage_percent reset to 80")
    target_coverage_percent = max(1, min(99, target_coverage_percent))

    interval_mode = str(payload.get("interval_mode", "symmetric")).strip().lower()
    if interval_mode not in {"symmetric", "custom", "multi_pack"}:
        interval_mode = "symmetric"
        warnings.append("interval_mode reset to symmetric")

    q_low_percent, q_high_percent, effective_coverage = _effective_symmetric_coverage(target_coverage_percent)
    implied_coverage = effective_coverage
    requested_coverage = target_coverage_percent / 100.0

    if interval_mode == "custom":
        try:
            q_low_percent = int(payload.get("custom_q_low_percent"))
            q_high_percent = int(payload.get("custom_q_high_percent"))
        except (TypeError, ValueError):
            errors.append("custom low/high percent is required for custom interval mode")
            q_low_percent, q_high_percent = 10, 90
        if q_low_percent >= q_high_percent:
            errors.append("custom q_low must be < q_high")
        q_low_percent = max(1, min(98, q_low_percent))
        q_high_percent = max(q_low_percent + 1, min(99, q_high_percent))
        effective_coverage = (q_high_percent - q_low_percent) / 100.0
        implied_coverage = effective_coverage

    multi_interval_keys: set[str] = set()
    multi_interval_coverages = _coerce_int_list(payload.get("multi_interval_coverages"), minimum=1, maximum=99)
    if interval_mode == "multi_pack":
        if not multi_interval_coverages:
            errors.append("multi_interval_coverages is required for multi_pack mode")
        for coverage_percent in multi_interval_coverages:
            pack_low, pack_high, _ = _effective_symmetric_coverage(coverage_percent)
            multi_interval_keys.add(_quantile_key_from_percent(pack_low))
            multi_interval_keys.add(_quantile_key_from_percent(pack_high))

    quantile_strategy = str(payload.get("quantile_strategy", "interval_only")).strip().lower()
    if quantile_strategy not in {"interval_only", "selected_set", "full_grid"}:
        quantile_strategy = "interval_only"
        warnings.append("quantile_strategy reset to interval_only")

    final_quantiles: set[str] = {
        _quantile_key_from_percent(q_low_percent),
        "q50",
        _quantile_key_from_percent(q_high_percent),
    }
    if interval_mode == "multi_pack":
        final_quantiles.update(multi_interval_keys)

    selected_quantiles = {
        key
        for key in (str(item).strip().lower() for item in payload.get("selected_quantiles", []))
        if _quantile_percent_from_key(key) is not None
    }
    try:
        quantile_soft_limit = int(payload.get("quantile_soft_limit", SOFT_QUANTILE_LIMIT))
    except (TypeError, ValueError):
        quantile_soft_limit = SOFT_QUANTILE_LIMIT
        warnings.append(f"quantile_soft_limit reset to {SOFT_QUANTILE_LIMIT}")
    quantile_soft_limit = max(1, min(99, quantile_soft_limit))
    if quantile_strategy == "selected_set":
        final_quantiles.update(selected_quantiles)
        if "q50" not in selected_quantiles:
            warnings.append("q50 was added automatically for compatibility")
        if _quantile_key_from_percent(q_low_percent) not in selected_quantiles:
            warnings.append("primary low quantile was added for legacy compatibility")
        if _quantile_key_from_percent(q_high_percent) not in selected_quantiles:
            warnings.append("primary high quantile was added for legacy compatibility")
    elif quantile_strategy == "full_grid":
        final_quantiles = {_quantile_key_from_percent(item) for item in range(1, 100)}
        warnings.append("full grid selected: resource heavy")

    final_quantiles_sorted = _sorted_quantile_keys(final_quantiles)
    requires_confirmation = len(final_quantiles_sorted) > quantile_soft_limit or quantile_strategy == "full_grid"
    if requires_confirmation:
        warnings.append("selected quantile set is resource heavy")

    if require_confirmation and requires_confirmation and not bool(payload.get("confirm_resource_heavy", False)):
        errors.append("resource-heavy quantile set requires explicit confirmation")

    hyperparams = _normalize_hyperparams(payload.get("hyperparams", {}), warnings=warnings)

    calibration_method = str(payload.get("calibration_method", "grid_scale")).strip().lower()
    if calibration_method != "grid_scale":
        warnings.append("conformal_cqr is not available yet; using grid_scale")
        calibration_method = "grid_scale"

    scale_grid = payload.get("scale_grid", DEFAULT_SCALE_GRID)
    if not isinstance(scale_grid, list):
        scale_grid = list(DEFAULT_SCALE_GRID)
    cleaned_scale_grid: list[float] = []
    for item in scale_grid:
        try:
            cleaned_scale_grid.append(float(item))
        except (TypeError, ValueError):
            continue
    if not cleaned_scale_grid:
        cleaned_scale_grid = list(DEFAULT_SCALE_GRID)
        warnings.append("scale_grid reset to defaults")

    feature_set_internal, feature_set_warning = _normalize_feature_set_version(payload.get("feature_set_version"))
    if feature_set_warning:
        warnings.append(feature_set_warning)

    steps_overrides_raw = dict(payload.get("steps_overrides", {}))
    steps_overrides: dict[str, int] = {}
    for horizon, step in steps_overrides_raw.items():
        try:
            value = int(step)
        except (TypeError, ValueError):
            continue
        if value > 0:
            steps_overrides[str(horizon)] = value

    sweep_dimensions = _normalized_sweep_dimensions(payload, warnings=warnings)
    sweep_variants_count = _estimate_sweep_variant_count_from_dimensions(payload, sweep_dimensions)
    enforce_sweep_rules = str(job_type or "").strip().lower() != "train_model"
    if enforce_sweep_rules and sweep_variants_count > SWEEP_VARIANT_CONFIRM_LIMIT:
        warnings.append(
            f"sweep expands to {sweep_variants_count} variants; explicit confirmation is recommended"
        )
    if enforce_sweep_rules and sweep_variants_count > SWEEP_VARIANT_HARD_LIMIT:
        errors.append(
            f"sweep expands to {sweep_variants_count} variants; hard limit is {SWEEP_VARIANT_HARD_LIMIT}"
        )
    if (
        enforce_sweep_rules
        and require_confirmation
        and sweep_variants_count > SWEEP_VARIANT_CONFIRM_LIMIT
        and not bool(payload.get("confirm_resource_heavy", False))
    ):
        errors.append(
            f"sweep with more than {SWEEP_VARIANT_CONFIRM_LIMIT} variants requires explicit confirmation"
        )

    requires_confirmation = bool(
        requires_confirmation
        or (enforce_sweep_rules and sweep_variants_count > SWEEP_VARIANT_CONFIRM_LIMIT)
    )

    response = {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "requested_coverage": requested_coverage,
        "effective_coverage": effective_coverage,
        "implied_coverage": implied_coverage,
        "q_low": _quantile_key_from_percent(q_low_percent),
        "q_high": _quantile_key_from_percent(q_high_percent),
        "quantile_soft_limit": quantile_soft_limit,
        "quantiles_final": final_quantiles_sorted,
        "quantiles_primary": [
            _quantile_key_from_percent(q_low_percent),
            "q50",
            _quantile_key_from_percent(q_high_percent),
        ],
        "quantiles_multi_interval": _sorted_quantile_keys(multi_interval_keys),
        "requires_confirmation": requires_confirmation,
        "sweep_variants_count": sweep_variants_count,
    }

    normalized = {
        "asset": asset,
        "training_budget_preset": training_budget_preset,
        "horizons": horizons,
        "base_timeframe_mode": base_timeframe_mode,
        "interval_mode": interval_mode,
        "steps_overrides": steps_overrides,
        "target_coverage_percent": target_coverage_percent,
        "requested_coverage": requested_coverage,
        "effective_coverage": effective_coverage,
        "q_low_percent": q_low_percent,
        "q_high_percent": q_high_percent,
        "q_low_key": _quantile_key_from_percent(q_low_percent),
        "q_high_key": _quantile_key_from_percent(q_high_percent),
        "multi_interval_coverages": multi_interval_coverages,
        "quantiles_final": final_quantiles_sorted,
        "quantile_strategy": quantile_strategy,
        "quantile_soft_limit": quantile_soft_limit,
        "train_window_mode": str(payload.get("train_window_mode", "expanding")).strip().lower(),
        "train_window_days": int(payload.get("train_window_days", 365) or 365),
        "val_days": int(payload.get("val_days", 30) or 30),
        "test_days": int(payload.get("test_days", 30) or 30),
        "walk_forward_enabled": bool(payload.get("walk_forward_enabled", False)),
        "wf_train_days": int(payload.get("wf_train_days", 180) or 180),
        "wf_val_days": int(payload.get("wf_val_days", 30) or 30),
        "wf_step_days": int(payload.get("wf_step_days", 30) or 30),
        "wf_folds": int(payload.get("wf_folds", 3) or 3),
        "feature_set_version": feature_set_internal,
        "feature_set_label": str(payload.get("feature_set_version", "feat_v1")).strip() or "feat_v1",
        "feature_groups": dict(payload.get("feature_groups", {})),
        "feature_overrides": dict(payload.get("feature_overrides", {})),
        "hyperparams": hyperparams,
        "calibration_method": calibration_method,
        "scale_grid": cleaned_scale_grid,
        "scale_selection_rule": str(payload.get("scale_selection_rule", "min_abs_coverage_gap_then_width")),
        "sweep_hyperparams": dict(sweep_dimensions.get("hyperparams", {})),
        "experiment_id": str(payload.get("experiment_id")).strip() if payload.get("experiment_id") else None,
        "parent_model_id": str(payload.get("parent_model_id")).strip() if payload.get("parent_model_id") else None,
        "notes": str(payload.get("notes")).strip() if payload.get("notes") else None,
        "tags": _normalize_tags(payload.get("tags", {})),
        "warnings": warnings,
        "errors": errors,
    }
    return BuilderValidation(response=response, normalized=normalized)


def _asset_symbol(config: dict[str, Any], asset: str) -> str:
    info = config.get("assets", {}).get(asset.upper())
    if not info:
        raise ValueError(f"asset not configured: {asset}")
    return str(info["symbol"])


def _parquet_folder(data_root: Path, layer: str, exchange: str, symbol: str, key: str) -> Path:
    return data_root / layer / exchange / symbol.replace("/", "_") / key


def _ensure_job_not_cancelled(store: Any, job_id: str) -> None:
    if store.is_training_job_cancel_requested(job_id):
        raise TrainingJobCancelled("training job cancelled")


def _load_base_1m_candles(
    *,
    config: dict[str, Any],
    data_root: Path,
    asset: str,
) -> pd.DataFrame:
    exchange = config.get("exchange", "binance")
    symbol = _asset_symbol(config, asset)
    clean_dir = _parquet_folder(data_root, "clean", exchange, symbol, "1m")
    candles = read_all_parquet(clean_dir)
    if candles.empty:
        raise ValueError(f"missing clean 1m candles for asset={asset}")
    return candles


def _prepare_base_1m_features(
    *,
    config: dict[str, Any],
    candles: pd.DataFrame,
) -> pd.DataFrame:
    return finalize_features(
        build_features(candles, config.get("feature_set_config", {})),
        warmup_drop=True,
    )


def _prepare_base_1m_labels(
    *,
    candles: pd.DataFrame,
    horizon: str,
    steps_ahead: int,
) -> pd.DataFrame:
    return build_labels(candles, horizon_name=horizon, steps_ahead=max(1, int(steps_ahead)))


def _prepare_legacy_dataset(
    *,
    config: dict[str, Any],
    data_root: Path,
    asset: str,
    horizon: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exchange = config.get("exchange", "binance")
    symbol = _asset_symbol(config, asset)
    timeframe = str(config.get("horizons_map", {}).get(horizon, {}).get("timeframe", horizon))
    features_dir = _parquet_folder(data_root, "features", exchange, symbol, timeframe)
    labels_dir = _parquet_folder(data_root, "labels", exchange, symbol, horizon)
    return read_all_parquet(features_dir), read_all_parquet(labels_dir)


def _format_variant_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _training_variant_signature(normalized: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(normalized.get("asset", "BTC")).upper(),
        tuple(str(item) for item in normalized.get("horizons", [])),
        str(normalized.get("base_timeframe_mode", "legacy")),
        str(normalized.get("interval_mode", "symmetric")),
        int(normalized.get("q_low_percent", 10)),
        int(normalized.get("q_high_percent", 90)),
        tuple(str(item) for item in normalized.get("quantiles_final", [])),
        str(normalized.get("train_window_mode", "expanding")),
        int(normalized.get("train_window_days", 365)),
        int(normalized.get("val_days", 30)),
        int(normalized.get("test_days", 30)),
        bool(normalized.get("walk_forward_enabled", False)),
        int(normalized.get("wf_train_days", 180)),
        int(normalized.get("wf_val_days", 30)),
        int(normalized.get("wf_step_days", 30)),
        int(normalized.get("wf_folds", 3)),
        str(normalized.get("feature_set_version", "v1")),
        tuple(sorted((str(key), bool(value)) for key, value in dict(normalized.get("feature_groups", {})).items())),
        tuple(
            sorted((str(key), repr(value)) for key, value in dict(normalized.get("feature_overrides", {})).items())
        ),
        tuple(sorted((str(key), repr(value)) for key, value in dict(normalized.get("hyperparams", {})).items())),
        str(normalized.get("calibration_method", "grid_scale")),
        tuple(round(float(item), 6) for item in normalized.get("scale_grid", [])),
        tuple(sorted((str(key), int(value)) for key, value in dict(normalized.get("steps_overrides", {})).items())),
    )


def _expand_training_variants(*, job_type: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_payload = deepcopy(payload)
    if job_type != "sweep_train":
        validation = validate_builder_payload(raw_payload, require_confirmation=True, job_type="train_model")
        if not validation.response["ok"]:
            raise ValueError("; ".join(validation.response["errors"]))
        return [
            {
                "payload": raw_payload,
                "validation": validation,
                "label": "variant 1/1",
            }
        ]

    dimensions = _normalized_sweep_dimensions(raw_payload)
    variants: list[dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    requested_total = _estimate_sweep_variant_count_from_dimensions(raw_payload, dimensions)
    hyperparam_axes = sorted(dict(dimensions.get("hyperparams", {})).items())

    for combo in product(
        dimensions["target_coverages"],
        dimensions["train_window_days"],
        dimensions["feature_set_versions"],
        dimensions["calibration_methods"],
        *[values for _, values in hyperparam_axes],
    ):
        coverage_value, train_window_days, feature_set_version, calibration_method, *hyperparam_values = combo
        variant_payload = deepcopy(raw_payload)
        variant_payload["target_coverage_percent"] = int(coverage_value)
        variant_payload["train_window_days"] = int(train_window_days)
        variant_payload["feature_set_version"] = str(feature_set_version)
        variant_payload["calibration_method"] = str(calibration_method)
        variant_payload["hyperparams"] = {
            **dict(raw_payload.get("hyperparams", {})),
        }
        hyperparam_label_parts: list[str] = []
        for index, (name, _) in enumerate(hyperparam_axes):
            value = hyperparam_values[index]
            variant_payload["hyperparams"][name] = value
            hyperparam_label_parts.append(f"{name}={_format_variant_value(value)}")
        validation = validate_builder_payload(variant_payload, require_confirmation=True, job_type="sweep_train")
        if not validation.response["ok"]:
            raise ValueError("; ".join(validation.response["errors"]))
        signature = _training_variant_signature(validation.normalized)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        if len(seen_signatures) > SWEEP_VARIANT_HARD_LIMIT:
            raise ValueError(
                f"sweep expands to more than {SWEEP_VARIANT_HARD_LIMIT} effective variants"
            )
        label_parts = [
            f"coverage={int(coverage_value)}",
            f"window={int(train_window_days)}",
            f"feature_set={feature_set_version}",
            f"calibration={calibration_method}",
            *hyperparam_label_parts,
        ]
        variants.append(
            {
                "payload": variant_payload,
                "validation": validation,
                "label": " ".join(label_parts),
            }
        )

    if not variants:
        raise ValueError("sweep produced no effective variants")

    total_effective = len(variants)
    for index, variant in enumerate(variants, start=1):
        variant["variant_id"] = f"v{index:03d}"
        variant["label"] = f"variant {index}/{total_effective} [{variant['label']}]"
        variant["requested_total"] = requested_total
        variant["effective_total"] = total_effective
    return variants


def _build_variant_config(base_config: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["feature_set"] = normalized["feature_set_version"]
    cfg["feature_set_config"] = _build_feature_config(cfg.get("feature_set_config", {}), normalized)
    cfg["training"] = {**cfg.get("training", {})}
    cfg["training"].update(
        {
            key: value
            for key, value in normalized["hyperparams"].items()
            if value not in (None, "")
        }
    )
    cfg["training"]["quantiles"] = [
        int(item[1:]) / 100.0
        for item in normalized["quantiles_final"]
    ]
    cfg["training"]["primary_low_quantile"] = normalized["q_low_percent"] / 100.0
    cfg["training"]["primary_high_quantile"] = normalized["q_high_percent"] / 100.0
    cfg["training"]["target_coverage"] = normalized["effective_coverage"]
    cfg["training"]["train_window_mode"] = normalized["train_window_mode"]
    cfg["training"]["train_window_days"] = normalized["train_window_days"]
    cfg["training"]["val_days"] = normalized["val_days"]
    cfg["training"]["test_days"] = normalized["test_days"]
    cfg["training"]["scale_grid"] = normalized["scale_grid"]
    if "seed" in normalized["hyperparams"] and normalized["hyperparams"]["seed"] not in (None, ""):
        cfg["seed"] = int(normalized["hyperparams"]["seed"])
    effective_num_boost_round = int(
        cfg["training"].get("num_boost_round", cfg["training"].get("n_estimators", 120))
    )
    effective_early_stopping_rounds = int(cfg["training"].get("early_stopping_rounds", 0) or 0)
    effective_scale_grid = list(cfg["training"].get("scale_grid", []) or [])
    selected_horizons = normalized["horizons"]

    cfg["builder_metadata"] = {
        "requested_coverage": normalized["requested_coverage"],
        "effective_coverage": normalized["effective_coverage"],
        "interval_mode": normalized["interval_mode"],
        "quantiles_final": normalized["quantiles_final"],
        "quantile_strategy": normalized["quantile_strategy"],
        "quantile_soft_limit": normalized["quantile_soft_limit"],
        "feature_set_label": normalized["feature_set_label"],
        "calibration_method": normalized["calibration_method"],
        "scale_grid": normalized["scale_grid"],
        "scale_selection_rule": normalized["scale_selection_rule"],
        "training_budget_preset": normalized.get("training_budget_preset", "custom"),
        "training_budget_effective": {
            "horizons": list(selected_horizons),
            "quantile_strategy": normalized["quantile_strategy"],
            "train_window_days": normalized["train_window_days"],
            "val_days": normalized["val_days"],
            "test_days": normalized["test_days"],
            "num_boost_round": effective_num_boost_round,
            "early_stopping_rounds": effective_early_stopping_rounds,
            "scale_grid": list(effective_scale_grid),
        },
        "experiment_id": normalized["experiment_id"],
        "parent_model_id": normalized["parent_model_id"],
        "notes": normalized["notes"],
        "tags": normalized["tags"],
        "base_timeframe_mode": normalized["base_timeframe_mode"],
        "train_window_mode": normalized["train_window_mode"],
        "train_window_days": normalized["train_window_days"],
        "val_days": normalized["val_days"],
        "test_days": normalized["test_days"],
        "walk_forward_enabled": normalized["walk_forward_enabled"],
        "walk_forward": {
            "train_days": normalized["wf_train_days"],
            "val_days": normalized["wf_val_days"],
            "step_days": normalized["wf_step_days"],
            "folds": normalized["wf_folds"],
        }
        if normalized["walk_forward_enabled"]
        else None,
        "steps_overrides": normalized["steps_overrides"],
    }
    if normalized["walk_forward_enabled"]:
        cfg["walk_forward"] = {
            **cfg.get("walk_forward", {}),
            "train_days": normalized["wf_train_days"],
            "val_days": normalized["wf_val_days"],
            "test_days": normalized["test_days"],
            "step_days": normalized["wf_step_days"],
            "max_splits": normalized["wf_folds"],
        }

    steps_runtime = {
        horizon: (
            normalized["steps_overrides"].get(horizon, AUTO_STEPS_AHEAD.get(horizon, 1))
            if normalized["base_timeframe_mode"] == "base_1m"
            else 1
        )
        for horizon in selected_horizons
    }
    cfg["builder_metadata"]["steps_runtime"] = steps_runtime
    cfg["horizons"] = list(selected_horizons)
    cfg["timeframes"] = [
        str(cfg.get("horizons_map", {}).get(horizon, {}).get("timeframe", horizon))
        for horizon in selected_horizons
    ]
    cfg["horizons_map"] = {
        horizon: dict(base_config.get("horizons_map", {}).get(horizon, {"timeframe": horizon}))
        for horizon in selected_horizons
    }
    return cfg


def run_walk_forward_job(
    *,
    job: dict[str, Any],
    store: Any,
    base_config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    payload = deepcopy(dict(job.get("params", {})))
    payload["walk_forward_enabled"] = True
    validation = validate_builder_payload(payload, require_confirmation=True, job_type="wf_eval")
    if not validation.response["ok"]:
        raise ValueError("; ".join(validation.response["errors"]))

    normalized = validation.normalized
    cfg = _build_variant_config(base_config, normalized)
    paths = get_runtime_paths(base_config, root=root)
    job_id = str(job["job_id"])
    selected_horizons = list(normalized["horizons"])
    total_horizons = max(1, len(selected_horizons))

    def set_stage(stage: str, progress: int, message: str) -> None:
        store.update_training_job(job_id, stage=stage, progress=progress)
        store.add_training_job_log(job_id, message=message, stage=stage)

    def set_progress(stage: str, progress: int) -> None:
        store.update_training_job(job_id, stage=stage, progress=progress)

    _ensure_job_not_cancelled(store, job_id)
    set_stage("data", 5, "checking cached market data for walk-forward evaluation")

    exchange = base_config.get("exchange", "binance")
    symbol = _asset_symbol(base_config, normalized["asset"])
    clean_1m_dir = _parquet_folder(paths.data_root, "clean", exchange, symbol, "1m")
    if not clean_1m_dir.exists() or read_all_parquet(clean_1m_dir).empty:
        set_stage("data", 12, "running data pipeline")
        run_data_pipeline(base_config, asset=normalized["asset"], root=paths.root)
    else:
        store.add_training_job_log(job_id, message="using cached clean 1m data", stage="data")

    if normalized["base_timeframe_mode"] == "legacy":
        set_stage("features", 20, "checking cached features / labels")
        missing = False
        for horizon in selected_horizons:
            features_df, labels_df = _prepare_legacy_dataset(
                config=cfg,
                data_root=paths.data_root,
                asset=normalized["asset"],
                horizon=horizon,
            )
            if features_df.empty or labels_df.empty:
                missing = True
                break
        if missing:
            set_stage("features", 28, "building features / labels")
            run_feature_label_pipeline(cfg, asset=normalized["asset"], root=paths.root)
        else:
            store.add_training_job_log(job_id, message="using cached features / labels", stage="features")
        base_1m_candles = None
        base_1m_features = None
    else:
        set_stage("features", 28, "building shared 1m features")
        base_1m_candles = _load_base_1m_candles(
            config=cfg,
            data_root=paths.data_root,
            asset=normalized["asset"],
        )
        base_1m_features = _prepare_base_1m_features(config=cfg, candles=base_1m_candles)
        store.add_training_job_log(
            job_id,
            message=f"shared 1m features ready for {len(selected_horizons)} horizon(s)",
            stage="features",
        )

    report_dir = paths.artifacts_root / "reports" / "walk_forward" / job_id
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    warnings: list[str] = [str(item) for item in validation.response.get("warnings", []) if str(item).strip()]

    for horizon_index, horizon in enumerate(selected_horizons, start=1):
        _ensure_job_not_cancelled(store, job_id)
        horizon_start = 35.0 + (50.0 * ((horizon_index - 1) / total_horizons))
        horizon_span = 50.0 / total_horizons
        set_stage(
            "train",
            int(round(horizon_start)),
            f"walk-forward evaluating horizon={horizon}",
        )

        if normalized["base_timeframe_mode"] == "legacy":
            features_df, labels_df = _prepare_legacy_dataset(
                config=cfg,
                data_root=paths.data_root,
                asset=normalized["asset"],
                horizon=horizon,
            )
        else:
            assert base_1m_candles is not None
            assert base_1m_features is not None
            features_df = base_1m_features
            labels_df = _prepare_base_1m_labels(
                candles=base_1m_candles,
                horizon=horizon,
                steps_ahead=normalized["steps_overrides"].get(horizon, AUTO_STEPS_AHEAD.get(horizon, 1)),
            )

        if features_df.empty or labels_df.empty:
            raise ValueError(f"no dataset available for horizon={horizon}")

        def progress_hook(_: str, ratio: float) -> None:
            bounded = max(0.0, min(1.0, float(ratio)))
            set_progress("train", int(round(horizon_start + (horizon_span * bounded))))

        report_payload = evaluate_walk_forward_horizon(
            asset=normalized["asset"],
            horizon=horizon,
            features_df=features_df,
            labels_df=labels_df,
            config=cfg,
            progress_hook=progress_hook,
            abort_if=lambda: bool(store.is_training_job_cancel_requested(job_id)),
        )
        report_path = report_dir / f"{normalized['asset'].lower()}_{horizon}_wf_report.json"
        dump_json(report_path, report_payload)
        reports.append(
            {
                "horizon": horizon,
                "report_path": str(report_path),
                "summary": dict(report_payload.get("summary", {})),
                "folds": list(report_payload.get("folds", [])),
            }
        )
        store.add_training_job_log(
            job_id,
            message=(
                f"walk-forward report saved: horizon={horizon}, "
                f"folds={int(report_payload.get('summary', {}).get('folds_evaluated', 0))}"
            ),
            stage="train",
        )

    _ensure_job_not_cancelled(store, job_id)
    set_stage("register", 95, "finalizing walk-forward reports")
    return {
        "warnings": warnings,
        "asset": normalized["asset"],
        "horizons": selected_horizons,
        "reports": reports,
        "reports_dir": str(report_dir),
        "folds_total": int(
            sum(int(item.get("summary", {}).get("folds_evaluated", 0)) for item in reports)
        ),
    }


def run_training_job(
    *,
    job: dict[str, Any],
    store: Any,
    base_config: dict[str, Any],
    root: str | Path,
    sync_models: Any,
) -> dict[str, Any]:
    job_type = str(job.get("type", "train_model"))
    variants = _expand_training_variants(job_type=job_type, payload=job.get("params", {}))
    first_normalized = variants[0]["validation"].normalized
    paths = get_runtime_paths(base_config, root=root)
    job_id = str(job["job_id"])
    total_variants = max(1, len(variants))
    overall_warnings: list[str] = []
    seen_warnings: set[str] = set()
    created_models: list[dict[str, Any]] = []
    if job_type == "sweep_train":
        requested_variants = int(variants[0].get("requested_total", total_variants))
        store.add_training_job_log(
            job_id,
            message=(
                f"sweep resolved to {total_variants} effective variant(s)"
                f" from {requested_variants} requested combination(s)"
            ),
            stage="queued",
        )

    def set_stage(stage: str, progress: int, message: str) -> None:
        store.update_training_job(job_id, stage=stage, progress=progress)
        store.add_training_job_log(job_id, message=message, stage=stage)

    def set_progress(stage: str, progress: int) -> None:
        store.update_training_job(job_id, stage=stage, progress=progress)

    def variant_progress(variant_index: int, local_fraction: float) -> int:
        start = 5.0 + (90.0 * ((variant_index - 1) / total_variants))
        span = 90.0 / total_variants
        bounded = max(0.0, min(1.0, float(local_fraction)))
        return max(1, min(95, int(round(start + (span * bounded)))))

    _ensure_job_not_cancelled(store, job_id)
    set_stage("data", 5, "checking cached market data")

    exchange = base_config.get("exchange", "binance")
    symbol = _asset_symbol(base_config, first_normalized["asset"])
    clean_1m_dir = _parquet_folder(paths.data_root, "clean", exchange, symbol, "1m")
    if not clean_1m_dir.exists() or read_all_parquet(clean_1m_dir).empty:
        set_stage("data", 10, "running data pipeline")
        run_data_pipeline(base_config, asset=first_normalized["asset"], root=paths.root)
    else:
        store.add_training_job_log(job_id, message="using cached clean 1m data", stage="data")

    for variant_index, variant in enumerate(variants, start=1):
        _ensure_job_not_cancelled(store, job_id)
        validation = variant["validation"]
        normalized = validation.normalized
        cfg = _build_variant_config(base_config, normalized)
        variant_id = str(variant.get("variant_id", f"v{variant_index:03d}"))
        for warning in validation.response.get("warnings", []):
            warning_text = str(warning).strip()
            if not warning_text or warning_text in seen_warnings:
                continue
            seen_warnings.add(warning_text)
            overall_warnings.append(warning_text)

        selected_horizons = normalized["horizons"]
        variant_label = f"{variant_id} {variant['label']}"
        quantile_count = max(1, len(normalized["quantiles_final"]))

        if normalized["base_timeframe_mode"] == "legacy":
            set_stage("features", variant_progress(variant_index, 0.10), f"{variant_label}: checking cached features / labels")
            missing = False
            for horizon in selected_horizons:
                features_df, labels_df = _prepare_legacy_dataset(
                    config=cfg,
                    data_root=paths.data_root,
                    asset=normalized["asset"],
                    horizon=horizon,
                )
                if features_df.empty or labels_df.empty:
                    missing = True
                    break
            if missing:
                set_stage(
                    "features",
                    variant_progress(variant_index, 0.18),
                    f"{variant_label}: building features / labels",
                )
                run_feature_label_pipeline(cfg, asset=normalized["asset"], root=paths.root)
            else:
                store.add_training_job_log(
                    job_id,
                    message=f"{variant_label}: using cached features / labels",
                    stage="features",
                )
            base_1m_candles = None
            base_1m_features = None
        else:
            set_stage(
                "features",
                variant_progress(variant_index, 0.14),
                f"{variant_label}: building shared 1m features",
            )
            base_1m_candles = _load_base_1m_candles(
                config=cfg,
                data_root=paths.data_root,
                asset=normalized["asset"],
            )
            base_1m_features = _prepare_base_1m_features(config=cfg, candles=base_1m_candles)
            store.add_training_job_log(
                job_id,
                message=f"{variant_label}: shared 1m features ready for {len(selected_horizons)} horizon(s)",
                stage="features",
            )

        total_horizons = max(1, len(selected_horizons))
        for horizon_index, horizon in enumerate(selected_horizons, start=1):
            _ensure_job_not_cancelled(store, job_id)
            set_stage(
                "train",
                variant_progress(variant_index, 0.25 + (((horizon_index - 1) / total_horizons) * 0.45)),
                f"{variant_label}: training horizon={horizon} ({quantile_count} q)",
            )

            if normalized["base_timeframe_mode"] == "legacy":
                features_df, labels_df = _prepare_legacy_dataset(
                    config=cfg,
                    data_root=paths.data_root,
                    asset=normalized["asset"],
                    horizon=horizon,
                )
            else:
                assert base_1m_candles is not None
                assert base_1m_features is not None
                features_df = base_1m_features
                labels_df = _prepare_base_1m_labels(
                    candles=base_1m_candles,
                    horizon=horizon,
                    steps_ahead=normalized["steps_overrides"].get(horizon, AUTO_STEPS_AHEAD.get(horizon, 1)),
                )

            if features_df.empty or labels_df.empty:
                raise ValueError(f"no dataset available for horizon={horizon}")

            def progress_hook(_: str, ratio: float) -> None:
                horizon_fraction = (horizon_index - 1 + max(0.0, min(1.0, float(ratio)))) / total_horizons
                set_progress(
                    "train",
                    variant_progress(variant_index, 0.25 + (0.45 * horizon_fraction)),
                )

            result = train_horizon_model(
                asset=normalized["asset"],
                horizon=horizon,
                features_df=features_df,
                labels_df=labels_df,
                config=cfg,
                artifacts_root=paths.artifacts_root,
                progress_hook=progress_hook,
                abort_if=lambda: bool(store.is_training_job_cancel_requested(job_id)),
            )
            created_models.append(
                {
                    "horizon": horizon,
                    "model_version": result["model_version"],
                    "variant_id": variant_id,
                    "trained_quantiles": result.get("trained_quantiles", []),
                    "primary_interval": result.get("primary_interval", {}),
                    "metrics_json": result.get("registry_metrics", {}),
                    "params_json": result.get("params_json", {}),
                    "wf_report_path": result.get("wf_report_path"),
                    "wf_report_summary": result.get("wf_report_summary", {}),
                    "variant_label": variant_label,
                }
            )
            store.upsert_model(
                {
                    "model_id": result["model_version"],
                    "asset": normalized["asset"],
                    "horizon": horizon,
                    "name": f"{normalized['asset']} {horizon}",
                    "created_at": result.get("created_at_utc"),
                    "git_commit": str(result.get("git_commit", "unknown")),
                    "train_time": result.get("created_at_utc"),
                    "dataset_hash": str(result.get("dataset_hash", "unknown")),
                    "config_hash": str(result.get("config_hash", "unknown")),
                    "metrics_json": result.get("registry_metrics", {}),
                    "params_json": result.get("params_json", {}),
                    "is_production": False,
                    "status": "active",
                    "notes": normalized["notes"],
                    "tags": {
                        **dict(normalized["tags"]),
                        "variant_id": variant_id,
                    },
                    "parent_model_id": normalized["parent_model_id"],
                    "experiment_id": normalized["experiment_id"],
                }
            )
            set_stage(
                "calibrate",
                variant_progress(variant_index, 0.76 + ((horizon_index / total_horizons) * 0.14)),
                f"{variant_label}: calibration saved for horizon={horizon}",
            )

    _ensure_job_not_cancelled(store, job_id)
    set_stage("register", 95, "syncing model registry")
    sync_models()
    store.add_training_job_log(job_id, message="model registry synced", stage="register")

    return {
        "warnings": overall_warnings,
        "model_ids": [str(item["model_version"]) for item in created_models],
        "models": created_models,
        "sweep_variants_count": total_variants,
        "sweep_requested_variants_count": int(variants[0].get("requested_total", total_variants)),
    }
