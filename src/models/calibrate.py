"""Calibration helpers for quantile intervals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable
from typing import Any

import numpy as np

_DEFAULT_SELECTION_RULE = "min_abs_coverage_gap_then_min_width"
_SELECTION_RULE_ALIASES = {
    "min_abs_coverage_gap": "min_abs_coverage_gap",
    "min_abs_coverage_gap_then_min_width": "min_abs_coverage_gap_then_min_width",
    "min_abs_coverage_gap_then_width": "min_abs_coverage_gap_then_min_width",
    "min_score": "min_score",
}


def _normalize_selection_rule(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    return _SELECTION_RULE_ALIASES.get(key, _DEFAULT_SELECTION_RULE)


def _clean_candidate_scales(candidate_scales: Iterable[float] | None) -> list[float]:
    if candidate_scales is None:
        return [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
    out: list[float] = []
    seen: set[float] = set()
    for item in candidate_scales:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if not isfinite(value) or value <= 0:
            continue
        signature = round(value, 12)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(value)
    if not out:
        return [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
    return out


@dataclass(frozen=True)
class IntervalCalibrator:
    target_coverage: float = 0.80
    spread_scale: float = 1.0
    selection_rule: str = _DEFAULT_SELECTION_RULE
    score_weight_gap: float = 1.0
    score_weight_width: float = 1.0
    scale_table: tuple[dict[str, float], ...] = ()

    def apply(self, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half_width = (q90 - q10) / 2.0
        center = q50
        low = center - self.spread_scale * half_width
        high = center + self.spread_scale * half_width
        return low, high

    @property
    def chosen_scale(self) -> float:
        return float(self.spread_scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_coverage": float(self.target_coverage),
            "spread_scale": float(self.spread_scale),
            "chosen_scale": float(self.spread_scale),
            "selection_rule": str(self.selection_rule),
            "score_weight_gap": float(self.score_weight_gap),
            "score_weight_width": float(self.score_weight_width),
            "scale_table": [dict(item) for item in self.scale_table],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntervalCalibrator":
        raw_table = payload.get("scale_table")
        scale_table: list[dict[str, float]] = []
        if isinstance(raw_table, list):
            for row in raw_table:
                if not isinstance(row, dict):
                    continue
                try:
                    scale_table.append(
                        {
                            "scale": float(row.get("scale", 1.0)),
                            "cov_val": float(row.get("cov_val", 0.0)),
                            "width_val": float(row.get("width_val", 0.0)),
                            "coverage_gap": float(row.get("coverage_gap", 0.0)),
                            "score_val": float(row.get("score_val", 0.0)),
                        }
                    )
                except (TypeError, ValueError):
                    continue
        return cls(
            target_coverage=float(payload.get("target_coverage", 0.8)),
            spread_scale=float(payload.get("chosen_scale", payload.get("spread_scale", 1.0))),
            selection_rule=_normalize_selection_rule(payload.get("selection_rule")),
            score_weight_gap=float(payload.get("score_weight_gap", 1.0)),
            score_weight_width=float(payload.get("score_weight_width", 1.0)),
            scale_table=tuple(scale_table),
        )


def fit_interval_calibrator(
    y_true: np.ndarray,
    q10_pred: np.ndarray,
    q50_pred: np.ndarray,
    q90_pred: np.ndarray,
    target_coverage: float = 0.80,
    candidate_scales: Iterable[float] | None = None,
    selection_rule: str = _DEFAULT_SELECTION_RULE,
    score_weight_gap: float = 1.0,
    score_weight_width: float = 1.0,
) -> IntervalCalibrator:
    if len(y_true) == 0:
        raise ValueError("y_true must contain at least one sample")

    scales = _clean_candidate_scales(candidate_scales)
    rule = _normalize_selection_rule(selection_rule)
    weight_gap = float(score_weight_gap)
    weight_width = float(score_weight_width)
    half_width = (q90_pred - q10_pred) / 2.0
    center = q50_pred
    table: list[dict[str, float]] = []
    ranked_rows: list[tuple[tuple[float, ...], dict[str, float]]] = []

    for idx, scale in enumerate(scales):
        low = center - scale * half_width
        high = center + scale * half_width
        coverage = float(np.mean((y_true >= low) & (y_true <= high)))
        width = float(np.mean(high - low))
        gap = float(abs(coverage - target_coverage))
        score = float((gap * weight_gap) + (width * weight_width))
        row = {
            "scale": float(scale),
            "cov_val": coverage,
            "width_val": width,
            "coverage_gap": gap,
            "score_val": score,
        }
        table.append(row)

        if rule == "min_abs_coverage_gap":
            rank_key = (gap, float(idx))
        elif rule == "min_score":
            rank_key = (score, float(idx))
        else:
            rank_key = (gap, width, float(idx))
        ranked_rows.append((rank_key, row))

    ranked_rows.sort(key=lambda item: item[0])
    best_row = ranked_rows[0][1]
    return IntervalCalibrator(
        target_coverage=float(target_coverage),
        spread_scale=float(best_row["scale"]),
        selection_rule=rule,
        score_weight_gap=weight_gap,
        score_weight_width=weight_width,
        scale_table=tuple(table),
    )

