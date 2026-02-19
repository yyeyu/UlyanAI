"""Calibration helpers for quantile intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IntervalCalibrator:
    target_coverage: float = 0.80
    spread_scale: float = 1.0

    def apply(self, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half_width = (q90 - q10) / 2.0
        center = q50
        low = center - self.spread_scale * half_width
        high = center + self.spread_scale * half_width
        return low, high

    def to_dict(self) -> dict[str, float]:
        return {"target_coverage": self.target_coverage, "spread_scale": self.spread_scale}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "IntervalCalibrator":
        return cls(
            target_coverage=float(payload.get("target_coverage", 0.8)),
            spread_scale=float(payload.get("spread_scale", 1.0)),
        )


def fit_interval_calibrator(
    y_true: np.ndarray,
    q10_pred: np.ndarray,
    q50_pred: np.ndarray,
    q90_pred: np.ndarray,
    target_coverage: float = 0.80,
    candidate_scales: Iterable[float] | None = None,
) -> IntervalCalibrator:
    if candidate_scales is None:
        candidate_scales = (0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0)

    best_scale = 1.0
    best_err = float("inf")

    for scale in candidate_scales:
        half_width = (q90_pred - q10_pred) / 2.0
        center = q50_pred
        low = center - scale * half_width
        high = center + scale * half_width
        coverage = float(np.mean((y_true >= low) & (y_true <= high)))
        err = abs(coverage - target_coverage)
        if err < best_err:
            best_err = err
            best_scale = float(scale)

    return IntervalCalibrator(target_coverage=target_coverage, spread_scale=best_scale)

