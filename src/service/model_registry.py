"""Helpers for registering trained model artifacts in the SQLite registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.service.event_store import EventStore


def _registry_metrics_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("registry_metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    return {
        "train": metadata.get("train_metrics_raw", {}),
        "val": metadata.get("val_metrics_raw", {}),
        "test": metadata.get("test_metrics_raw", {}),
        "calibrated": metadata.get("test_metrics_calibrated", {}),
    }


def _fallback_params_from_metadata(metadata: dict[str, Any], *, asset: str, horizon: str) -> dict[str, Any]:
    base_timeframe_mode = str(metadata.get("base_timeframe_mode", "legacy")).strip().lower()
    if base_timeframe_mode == "base_1m":
        base_timeframe = "1m"
    else:
        base_timeframe = str(metadata.get("base_timeframe", horizon))
    steps_ahead = metadata.get("steps_ahead")
    try:
        parsed_steps = int(steps_ahead) if steps_ahead is not None else None
    except (TypeError, ValueError):
        parsed_steps = None
    return {
        "asset": asset.upper(),
        "horizon": horizon,
        "base_timeframe": base_timeframe,
        "steps_ahead": parsed_steps,
        "quantiles": list(metadata.get("trained_quantiles", [])),
        "target_coverage_requested": float(metadata.get("requested_coverage", metadata.get("effective_coverage", 0.8))),
        "target_coverage_effective": float(metadata.get("effective_coverage", metadata.get("requested_coverage", 0.8))),
        "interval_mode": str(metadata.get("interval_mode", "symmetric")),
        "train_window_mode": str(metadata.get("train_window_mode", "expanding")),
        "train_window_days": metadata.get("train_window_days"),
        "val_days": metadata.get("val_days"),
        "test_days": metadata.get("test_days"),
        "walk_forward": metadata.get("walk_forward"),
        "feature_set_version": str(metadata.get("feature_set_label", metadata.get("feature_set", "feat_v1"))),
        "calibration": {
            "method": str(metadata.get("calibration_method", "grid_scale")),
            "scale_grid": list(metadata.get("scale_grid", [])),
            "selection_rule": str(metadata.get("scale_selection_rule", "min_abs_coverage_gap_then_width")),
            "selected_scale": metadata.get("calibrator", {}).get("scale"),
            "target_coverage": metadata.get("calibrator", {}).get("target_coverage"),
        },
        "seed": metadata.get("seed"),
        "data_range_start": metadata.get("data_range_start"),
        "data_range_end": metadata.get("data_range_end"),
    }


def sync_models_registry(*, store: EventStore, artifacts_root: Path) -> None:
    models_root = artifacts_root / "models"
    if not models_root.exists():
        return
    for asset_dir in sorted([path for path in models_root.iterdir() if path.is_dir()]):
        asset = asset_dir.name.upper()
        for horizon_dir in sorted([path for path in asset_dir.iterdir() if path.is_dir()]):
            horizon = horizon_dir.name
            versions = sorted([path for path in horizon_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
            if not versions:
                continue
            latest_model_id = versions[-1].name
            for model_dir in versions:
                metadata_path = model_dir / "metadata.json"
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except Exception:
                        metadata = {}
                model_id = str(metadata.get("model_version", model_dir.name))
                if model_dir == versions[-1]:
                    latest_model_id = model_id
                params_json = metadata.get("params_json")
                if not isinstance(params_json, dict) or not params_json:
                    params_json = _fallback_params_from_metadata(metadata, asset=asset, horizon=horizon)
                store.upsert_model(
                    {
                        "model_id": model_id,
                        "asset": asset,
                        "horizon": horizon,
                        "name": f"{asset} {horizon}",
                        "created_at": str(metadata.get("created_at_utc", datetime.now(timezone.utc).isoformat())),
                        "git_commit": str(metadata.get("git_commit", "unknown")),
                        "train_time": str(metadata.get("created_at_utc", "unknown")),
                        "dataset_hash": str(metadata.get("dataset_hash", "unknown")),
                        "config_hash": str(metadata.get("config_hash", "unknown")),
                        "metrics_json": _registry_metrics_from_metadata(metadata),
                        "params_json": params_json,
                        "is_production": model_dir == versions[-1],
                        "status": "active",
                        "notes": metadata.get("notes"),
                        "tags": metadata.get("tags", {}),
                        "parent_model_id": metadata.get("parent_model_id"),
                        "experiment_id": metadata.get("experiment_id"),
                    }
                )
            store.set_production_model(asset=asset, horizon=horizon, model_id=latest_model_id)
