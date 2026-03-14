"""Prediction helpers for converting return-quantiles to price ranges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any

import numpy as np

from src.models.calibrate import IntervalCalibrator
from src.models.lgbm_compat import load_booster_compat
from src.models.registry import latest_model_dir, read_metadata
from src.utils import load_json

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - import guarded for lightweight environments.
    lgb = None


def _quantile_key(alpha: float) -> str:
    pct = int(round(float(alpha) * 100))
    return f"q{pct:02d}"


def _quantile_sort_key(key: str) -> tuple[int, str]:
    suffix = key[1:] if key.startswith("q") else key
    try:
        return int(suffix), key
    except ValueError:
        return 999, key


@dataclass(frozen=True)
class ReturnQuantiles:
    """Internal model output in return space."""

    q10: float
    q50: float
    q90: float


@dataclass(frozen=True)
class PriceLevels:
    """External product output in price space."""

    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class BundlePrediction:
    primary: ReturnQuantiles
    extra_quantiles: dict[str, float]


class NonMonotonicPriceError(ValueError):
    """Raised when p10 <= p50 <= p90 does not hold."""


def _ensure_monotonic(p10: float, p50: float, p90: float) -> None:
    if not (p10 <= p50 <= p90):
        raise NonMonotonicPriceError(
            f"non-monotonic price levels: p10={p10}, p50={p50}, p90={p90}"
        )


def returns_to_price_levels(
    *,
    price_spot: float,
    q10_return: float,
    q50_return: float,
    q90_return: float,
) -> PriceLevels:
    """Convert return quantiles to price quantiles with monotonicity check."""
    if price_spot <= 0:
        raise ValueError("price_spot must be positive")

    p10 = price_spot * exp(q10_return)
    p50 = price_spot * exp(q50_return)
    p90 = price_spot * exp(q90_return)
    _ensure_monotonic(p10, p50, p90)
    return PriceLevels(p10=p10, p50=p50, p90=p90)


def build_price_prediction(
    *,
    asset: str,
    horizon: str,
    as_of: datetime | None,
    price_spot: float,
    return_quantiles: ReturnQuantiles,
    nominal: float = 0.80,
    calibration_score: float = 1.0,
    drift_flag: bool = False,
    model_version: str = "unknown",
    stale_data: bool = False,
    extra_return_quantiles: dict[str, float] | None = None,
) -> dict:
    """Build external prediction payload in price-range language only."""
    levels = returns_to_price_levels(
        price_spot=price_spot,
        q10_return=return_quantiles.q10,
        q50_return=return_quantiles.q50,
        q90_return=return_quantiles.q90,
    )

    as_of_utc = as_of or datetime.now(timezone.utc)
    if as_of_utc.tzinfo is None:
        as_of_utc = as_of_utc.replace(tzinfo=timezone.utc)
    as_of_utc = as_of_utc.astimezone(timezone.utc)

    payload = {
        "asset": asset.upper(),
        "horizon": horizon,
        "as_of": as_of_utc.isoformat(),
        "price_spot": round(price_spot, 8),
        "mode": "price_ranges",
        "price_levels": {
            "p10": round(levels.p10, 8),
            "p50": round(levels.p50, 8),
            "p90": round(levels.p90, 8),
        },
        "price_range": {
            "low": round(levels.p10, 8),
            "high": round(levels.p90, 8),
            "nominal": nominal,
        },
        "median_price": round(levels.p50, 8),
        "confidence": {
            "calibration_score": calibration_score,
            "drift_flag": drift_flag,
        },
        "model_version": model_version,
        "stale_data": stale_data,
    }
    if extra_return_quantiles:
        payload["quantiles_pred"] = {
            key: round(price_spot * exp(float(value)), 8)
            for key, value in sorted(extra_return_quantiles.items(), key=lambda item: _quantile_sort_key(item[0]))
        }
    else:
        payload["quantiles_pred"] = {}
    return payload


@dataclass
class ModelBundle:
    models: dict[str, Any]
    calibrator: IntervalCalibrator
    metadata: dict[str, Any]
    quantile_keys: list[str]
    primary_low_key: str
    primary_mid_key: str
    primary_high_key: str


def _bundle_quantile_keys(model_path: Path, metadata: dict[str, Any]) -> list[str]:
    keys = metadata.get("trained_quantiles")
    if isinstance(keys, list):
        out = [str(item) for item in keys if str(item).startswith("q")]
        if out:
            return sorted(set(out), key=_quantile_sort_key)

    files = sorted(model_path.glob("model_q*.txt"))
    if files:
        out = []
        for file in files:
            stem = file.stem
            if stem.startswith("model_q"):
                out.append(stem.replace("model_", "", 1))
        if out:
            return sorted(set(out), key=_quantile_sort_key)
    return ["q10", "q50", "q90"]


def load_model_bundle(
    artifacts_root: Path,
    asset: str,
    horizon: str,
    model_version: str | None = None,
) -> ModelBundle | None:
    if lgb is None:
        return None
    model_path = (
        latest_model_dir(artifacts_root, asset, horizon)
        if model_version is None
        else artifacts_root / "models" / asset.upper() / horizon / model_version
    )
    if model_path is None or not model_path.exists():
        return None

    calib_path = model_path / "calibrator.json"
    metadata_path = model_path / "metadata.json"
    if not calib_path.exists():
        return None

    metadata = read_metadata(model_path) if metadata_path.exists() else {}
    quantile_keys = _bundle_quantile_keys(model_path, metadata)
    models: dict[str, Any] = {}
    for key in quantile_keys:
        model_file = model_path / f"model_{key}.txt"
        if not model_file.exists():
            return None
        models[key] = load_booster_compat(lgb, model_file)

    primary_low_key = _quantile_key(float(metadata.get("primary_low_quantile", 0.1)))
    primary_mid_key = _quantile_key(float(metadata.get("primary_mid_quantile", 0.5)))
    primary_high_key = _quantile_key(float(metadata.get("primary_high_quantile", 0.9)))
    if primary_low_key not in models:
        primary_low_key = "q10" if "q10" in models else quantile_keys[0]
    if primary_mid_key not in models:
        primary_mid_key = "q50" if "q50" in models else quantile_keys[len(quantile_keys) // 2]
    if primary_high_key not in models:
        primary_high_key = "q90" if "q90" in models else quantile_keys[-1]

    calibrator = IntervalCalibrator.from_dict(load_json(calib_path))
    return ModelBundle(
        models=models,
        calibrator=calibrator,
        metadata=metadata,
        quantile_keys=quantile_keys,
        primary_low_key=primary_low_key,
        primary_mid_key=primary_mid_key,
        primary_high_key=primary_high_key,
    )


def predict_bundle_outputs(bundle: ModelBundle, feature_row: dict[str, float]) -> BundlePrediction:
    feature_cols = bundle.metadata.get("feature_columns", [])
    if not feature_cols:
        raise ValueError("metadata.feature_columns is required for inference")

    row = np.array([[float(feature_row[col]) for col in feature_cols]], dtype=float)
    raw_predictions = {
        key: float(model.predict(row)[0])
        for key, model in bundle.models.items()
    }

    low_cal, high_cal = bundle.calibrator.apply(
        q10=np.array([raw_predictions[bundle.primary_low_key]]),
        q50=np.array([raw_predictions[bundle.primary_mid_key]]),
        q90=np.array([raw_predictions[bundle.primary_high_key]]),
    )
    raw_predictions[bundle.primary_low_key] = float(low_cal[0])
    raw_predictions[bundle.primary_high_key] = float(high_cal[0])

    return BundlePrediction(
        primary=ReturnQuantiles(
            q10=raw_predictions[bundle.primary_low_key],
            q50=raw_predictions[bundle.primary_mid_key],
            q90=raw_predictions[bundle.primary_high_key],
        ),
        extra_quantiles={
            key: raw_predictions[key]
            for key in sorted(raw_predictions.keys(), key=_quantile_sort_key)
        },
    )


def predict_return_quantiles(
    bundle: ModelBundle,
    feature_row: dict[str, float],
) -> ReturnQuantiles:
    return predict_bundle_outputs(bundle, feature_row).primary
