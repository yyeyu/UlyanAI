"""Prediction helpers for converting return-quantiles to price ranges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any

import numpy as np

from src.models.calibrate import IntervalCalibrator
from src.models.registry import latest_model_dir, read_metadata
from src.utils import load_json

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - import guarded for lightweight environments.
    lgb = None


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

    return {
        "asset": asset.upper(),
        "horizon": horizon,
        "as_of": as_of_utc.isoformat().replace("+00:00", "Z"),
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


@dataclass
class ModelBundle:
    models: dict[str, Any]
    calibrator: IntervalCalibrator
    metadata: dict[str, Any]


def load_model_bundle(artifacts_root: Path, asset: str, horizon: str) -> ModelBundle | None:
    if lgb is None:
        return None
    model_path = latest_model_dir(artifacts_root, asset, horizon)
    if model_path is None:
        return None

    q10_path = model_path / "model_q10.txt"
    q50_path = model_path / "model_q50.txt"
    q90_path = model_path / "model_q90.txt"
    calib_path = model_path / "calibrator.json"
    metadata_path = model_path / "metadata.json"
    if not (q10_path.exists() and q50_path.exists() and q90_path.exists() and calib_path.exists()):
        return None

    models = {
        "q10": lgb.Booster(model_file=str(q10_path)),
        "q50": lgb.Booster(model_file=str(q50_path)),
        "q90": lgb.Booster(model_file=str(q90_path)),
    }
    calibrator = IntervalCalibrator.from_dict(load_json(calib_path))
    metadata = read_metadata(model_path) if metadata_path.exists() else {}
    return ModelBundle(models=models, calibrator=calibrator, metadata=metadata)


def predict_return_quantiles(
    bundle: ModelBundle,
    feature_row: dict[str, float],
) -> ReturnQuantiles:
    feature_cols = bundle.metadata.get("feature_columns", [])
    if not feature_cols:
        raise ValueError("metadata.feature_columns is required for inference")

    row = np.array([[float(feature_row[col]) for col in feature_cols]], dtype=float)
    q10 = float(bundle.models["q10"].predict(row)[0])
    q50 = float(bundle.models["q50"].predict(row)[0])
    q90 = float(bundle.models["q90"].predict(row)[0])
    low_cal, high_cal = bundle.calibrator.apply(
        q10=np.array([q10]),
        q50=np.array([q50]),
        q90=np.array([q90]),
    )
    return ReturnQuantiles(q10=float(low_cal[0]), q50=q50, q90=float(high_cal[0]))
