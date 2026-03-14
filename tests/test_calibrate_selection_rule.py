from __future__ import annotations

import numpy as np

from src.models.calibrate import IntervalCalibrator, fit_interval_calibrator


def _base_predictions(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(size, -1.0, dtype=float),
        np.zeros(size, dtype=float),
        np.full(size, 1.0, dtype=float),
    )


def test_selection_rule_tie_breaker_prefers_min_width() -> None:
    y_true = np.array([0.0, 0.1, -0.1], dtype=float)
    q10, q50, q90 = _base_predictions(len(y_true))
    scales = [2.0, 1.0]

    gap_only = fit_interval_calibrator(
        y_true=y_true,
        q10_pred=q10,
        q50_pred=q50,
        q90_pred=q90,
        target_coverage=1.0,
        candidate_scales=scales,
        selection_rule="min_abs_coverage_gap",
    )
    tie_with_width = fit_interval_calibrator(
        y_true=y_true,
        q10_pred=q10,
        q50_pred=q50,
        q90_pred=q90,
        target_coverage=1.0,
        candidate_scales=scales,
        selection_rule="min_abs_coverage_gap_then_min_width",
    )
    legacy_alias = fit_interval_calibrator(
        y_true=y_true,
        q10_pred=q10,
        q50_pred=q50,
        q90_pred=q90,
        target_coverage=1.0,
        candidate_scales=scales,
        selection_rule="min_abs_coverage_gap_then_width",
    )

    assert abs(gap_only.spread_scale - 2.0) < 1e-12
    assert abs(tie_with_width.spread_scale - 1.0) < 1e-12
    assert abs(legacy_alias.spread_scale - tie_with_width.spread_scale) < 1e-12


def test_min_score_rule_and_calibrator_payload_table() -> None:
    y_true = np.array([0.0, 0.6, -0.6], dtype=float)
    q10, q50, q90 = _base_predictions(len(y_true))
    scales = [0.5, 1.0]

    width_priority = fit_interval_calibrator(
        y_true=y_true,
        q10_pred=q10,
        q50_pred=q50,
        q90_pred=q90,
        target_coverage=1.0,
        candidate_scales=scales,
        selection_rule="min_score",
        score_weight_gap=0.1,
        score_weight_width=1.0,
    )
    gap_priority = fit_interval_calibrator(
        y_true=y_true,
        q10_pred=q10,
        q50_pred=q50,
        q90_pred=q90,
        target_coverage=1.0,
        candidate_scales=scales,
        selection_rule="min_score",
        score_weight_gap=10.0,
        score_weight_width=1.0,
    )

    assert abs(width_priority.spread_scale - 0.5) < 1e-12
    assert abs(gap_priority.spread_scale - 1.0) < 1e-12

    payload = gap_priority.to_dict()
    assert "chosen_scale" in payload
    assert abs(float(payload["chosen_scale"]) - float(payload["spread_scale"])) < 1e-12
    assert isinstance(payload.get("scale_table"), list)
    assert len(payload["scale_table"]) == 2
    for row in payload["scale_table"]:
        assert {"scale", "cov_val", "width_val", "coverage_gap", "score_val"} <= set(row.keys())

    restored = IntervalCalibrator.from_dict(payload)
    assert restored.selection_rule == "min_score"
    assert len(restored.scale_table) == 2
