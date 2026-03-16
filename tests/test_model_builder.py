from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import train_horizon_model
from src.service.event_store import EventStore
from src.service.model_builder import (
    _build_variant_config,
    _expand_training_variants,
    estimate_sweep_variant_count,
    run_training_job,
    validate_builder_payload,
)


class ModelBuilderSweepTests(unittest.TestCase):
    def test_generic_sweep_axes_preview_reports_breakdown_and_dedupe(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_axes": [
                {"path": "target_coverage_percent", "mode": "list", "values": [80, 81, 82]},
                {"path": "feature_groups.rsi", "mode": "list", "values": [True, False]},
                {"path": "hyperparams.num_leaves", "mode": "range", "start": 31, "end": 63, "step": 32},
            ],
        }

        validation = validate_builder_payload(payload, require_confirmation=False, job_type="sweep_train")

        self.assertTrue(validation.response["ok"])
        preview = validation.response["sweep_preview"]
        self.assertEqual(preview["requested_total"], 12)
        self.assertEqual(preview["effective_total"], 8)
        self.assertEqual(preview["duplicate_count"], 4)
        self.assertEqual(preview["estimated_model_count"], 8)
        self.assertEqual(validation.response["sweep_variants_count"], 8)
        axis_preview = {str(item["path"]): item for item in preview["axes"]}
        self.assertIn("target_coverage_percent", axis_preview)
        self.assertEqual(int(axis_preview["target_coverage_percent"]["requested_count"]), 3)
        self.assertIn("feature_groups.rsi", axis_preview)
        self.assertIn("hyperparams.num_leaves", axis_preview)

    def test_generic_sweep_axes_support_enum_bool_and_numeric_variants(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_axes": [
                {"path": "base_timeframe_mode", "mode": "list", "values": ["legacy", "base_1m"]},
                {"path": "feature_groups.rsi", "mode": "list", "values": [True, False]},
                {"path": "hyperparams.learning_rate", "mode": "range", "start": 0.02, "end": 0.05, "step": 0.03},
            ],
            "confirm_resource_heavy": True,
        }

        variants = _expand_training_variants(job_type="sweep_train", payload=payload)

        self.assertEqual(len(variants), 8)
        labels = [str(item["label"]) for item in variants]
        self.assertTrue(any("base_timeframe_mode=base_1m" in label for label in labels))
        self.assertTrue(any("feature_groups.rsi=false" in label for label in labels))
        self.assertTrue(any("hyperparams.learning_rate=0.02" in label for label in labels))

    def test_validate_builder_reports_sweep_variant_count(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_target_coverages": [80, 81, 82],
            "sweep_train_window_days": [180, 365],
            "sweep_feature_set_versions": ["feat_v1"],
            "sweep_calibration_methods": ["grid_scale"],
        }

        validation = validate_builder_payload(payload, require_confirmation=False)

        self.assertTrue(validation.response["ok"])
        self.assertEqual(validation.response["sweep_variants_count"], 4)
        self.assertEqual(estimate_sweep_variant_count(payload), 4)

    def test_expand_training_variants_dedupes_effectively_identical_sweep_variants(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "interval_mode": "custom",
            "custom_q_low_percent": 10,
            "custom_q_high_percent": 90,
            "sweep_target_coverages": [80, 90],
            "sweep_feature_set_versions": ["feat_v1", "feat_v2"],
            "sweep_calibration_methods": ["grid_scale", "conformal_cqr"],
        }

        variants = _expand_training_variants(job_type="sweep_train", payload=payload)

        self.assertEqual(len(variants), 1)
        self.assertIn("variant 1/1", str(variants[0]["label"]))

    def test_expand_training_variants_supports_hyperparam_cartesian_product(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_hyperparams": {
                "learning_rate": [0.02, 0.05],
                "num_leaves": [31, 63],
            },
        }

        variants = _expand_training_variants(job_type="sweep_train", payload=payload)

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            [str(item["variant_id"]) for item in variants],
            ["v001", "v002", "v003", "v004"],
        )
        labels = [str(item["label"]) for item in variants]
        self.assertTrue(any("learning_rate=0.02" in label for label in labels))
        self.assertTrue(any("num_leaves=63" in label for label in labels))

    def test_validate_builder_rejects_sweep_over_hard_limit(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_hyperparams": {
                "num_leaves": list(range(31, 241)),
            },
            "confirm_resource_heavy": True,
        }

        validation = validate_builder_payload(payload, require_confirmation=True)

        self.assertFalse(validation.response["ok"])
        self.assertTrue(
            any("hard limit" in str(item).lower() for item in validation.response["errors"])
        )

    def test_validate_builder_rejects_generic_sweep_over_hard_limit(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_axes": [
                {
                    "path": "hyperparams.num_leaves",
                    "mode": "list",
                    "values": list(range(31, 241)),
                }
            ],
            "confirm_resource_heavy": True,
        }

        validation = validate_builder_payload(payload, require_confirmation=True, job_type="sweep_train")

        self.assertFalse(validation.response["ok"])
        self.assertTrue(any("hard limit" in str(item).lower() for item in validation.response["errors"]))

    def test_validate_builder_allows_plain_train_even_if_sweep_axes_are_large(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_hyperparams": {
                "num_leaves": list(range(31, 241)),
            },
        }

        validation = validate_builder_payload(payload, require_confirmation=True, job_type="train_model")

        self.assertTrue(validation.response["ok"])

    def test_validate_builder_rejects_unsupported_sweep_axis_path(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_axes": [
                {"path": "hyperparams.unknown_knob", "mode": "list", "values": [1, 2]},
            ],
        }

        validation = validate_builder_payload(payload, require_confirmation=False, job_type="sweep_train")

        self.assertFalse(validation.response["ok"])
        self.assertTrue(any("unsupported sweep axis path" in str(item).lower() for item in validation.response["errors"]))

    def test_validate_builder_rejects_unimplemented_sweep_method_axis(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_axes": [
                {"path": "calibration_method", "mode": "list", "values": ["grid_scale", "conformal_cqr"]},
            ],
        }

        validation = validate_builder_payload(payload, require_confirmation=False, job_type="sweep_train")

        self.assertFalse(validation.response["ok"])
        self.assertTrue(any("not implemented" in str(item).lower() for item in validation.response["errors"]))

    def test_validate_builder_keeps_training_budget_preset(self) -> None:
        payload = {
            "asset": "BTC",
            "training_budget_preset": "F2",
            "horizons": ["5m", "15m", "1h"],
            "quantile_strategy": "full_grid",
        }

        validation = validate_builder_payload(payload, require_confirmation=False, job_type="train_model")

        self.assertTrue(validation.response["ok"])
        self.assertEqual(validation.normalized["training_budget_preset"], "F2")
        self.assertTrue(validation.response["requires_confirmation"])

    def test_build_variant_config_writes_training_budget_metadata(self) -> None:
        payload = {
            "asset": "BTC",
            "training_budget_preset": "F0",
            "horizons": ["5m"],
            "quantile_strategy": "interval_only",
            "train_window_days": 120,
            "val_days": 14,
            "test_days": 14,
            "hyperparams": {"num_boost_round": 60, "early_stopping_rounds": 10},
            "scale_grid": [0.8, 1.0, 1.2],
        }
        validation = validate_builder_payload(payload, require_confirmation=False, job_type="train_model")
        self.assertTrue(validation.response["ok"])

        cfg = _build_variant_config(load_config("configs"), validation.normalized)
        metadata = cfg.get("builder_metadata", {})
        self.assertEqual(metadata.get("training_budget_preset"), "F0")
        effective = metadata.get("training_budget_effective", {})
        self.assertEqual(effective.get("horizons"), ["5m"])
        self.assertEqual(effective.get("quantile_strategy"), "interval_only")
        self.assertEqual(int(effective.get("train_window_days", 0)), 120)
        self.assertEqual(int(effective.get("val_days", 0)), 14)
        self.assertEqual(int(effective.get("test_days", 0)), 14)
        self.assertEqual(int(effective.get("num_boost_round", 0)), 60)
        self.assertEqual(int(effective.get("early_stopping_rounds", 0)), 10)
        self.assertEqual(list(effective.get("scale_grid", [])), [0.8, 1.0, 1.2])

    def test_train_horizon_model_metadata_contains_budget_preset_and_effective_values(self) -> None:
        cfg = load_config("configs")
        cfg["training"]["num_boost_round"] = 5
        cfg["training"]["early_stopping_rounds"] = 2
        cfg["builder_metadata"] = {
            "training_budget_preset": "F1",
            "training_budget_effective": {
                "horizons": ["5m", "15m"],
                "quantile_strategy": "selected_set",
                "train_window_days": 365,
                "val_days": 30,
                "test_days": 30,
                "num_boost_round": 5,
                "early_stopping_rounds": 2,
                "scale_grid": [0.75, 1.0, 1.2, 1.5, 2.0],
            },
        }
        n = 120
        ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        close = 50000.0 + np.linspace(0.0, 200.0, n)
        features = pd.DataFrame(
            {
                "ts_utc": ts,
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "close": close,
                "f1": np.sin(np.linspace(0, 6.28, n)),
                "f2": np.cos(np.linspace(0, 6.28, n)),
            }
        )
        labels = pd.DataFrame(
            {
                "ts_utc": ts,
                "target_r_5m": np.linspace(-0.01, 0.01, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train_horizon_model(
                asset="BTC",
                horizon="5m",
                features_df=features,
                labels_df=labels,
                config=cfg,
                artifacts_root=Path(tmp_dir),
            )
            metadata_path = Path(result["artifact_dir"]) / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("training_budget_preset"), "F1")
            effective = payload.get("training_budget_effective", {})
            self.assertEqual(effective.get("quantile_strategy"), "selected_set")
            self.assertEqual(int(effective.get("num_boost_round", 0)), 5)
            self.assertEqual(int(effective.get("early_stopping_rounds", 0)), 2)
            self.assertEqual(list(effective.get("scale_grid", [])), [0.75, 1.0, 1.2, 1.5, 2.0])

    def test_sweep_hyperparams_create_four_variants_with_logs_and_variant_tags(self) -> None:
        payload = {
            "asset": "BTC",
            "horizons": ["5m"],
            "sweep_hyperparams": {
                "learning_rate": [0.02, 0.05],
                "num_leaves": [31, 63],
            },
            "confirm_resource_heavy": True,
        }
        features = pd.DataFrame(
            {
                "ts_utc": [pd.Timestamp("2026-01-01T00:00:00+00:00")],
                "f1": [1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "ts_utc": [pd.Timestamp("2026-01-01T00:00:00+00:00")],
                "target_r_5m": [0.001],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                job = store.create_job(job_type="sweep_train", params=payload)
                job_id = str(job["job_id"])
                cfg = load_config("configs")

                def _fake_train_horizon_model(*, asset, horizon, features_df, labels_df, config, artifacts_root, progress_hook, abort_if):
                    if progress_hook:
                        progress_hook("train", 1.0)
                    learning_rate = str(config.get("training", {}).get("learning_rate", "x")).replace(".", "p")
                    num_leaves = int(config.get("training", {}).get("num_leaves", 0))
                    model_version = f"{asset.lower()}_{horizon}_lr{learning_rate}_nl{num_leaves}"
                    return {
                        "model_version": model_version,
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "trained_quantiles": ["q10", "q50", "q90"],
                        "primary_interval": {"low": "q10", "mid": "q50", "high": "q90"},
                        "registry_metrics": {"test": {"coverage_return": 0.8}},
                        "params_json": {"asset": asset, "horizon": horizon, "base_timeframe": "5m"},
                        "git_commit": "test",
                        "dataset_hash": "dataset",
                        "config_hash": "config",
                    }

                with (
                    patch("src.service.model_builder.run_data_pipeline", return_value={}),
                    patch("src.service.model_builder._prepare_legacy_dataset", return_value=(features, labels)),
                    patch("src.service.model_builder.run_feature_label_pipeline", return_value={}),
                    patch("src.service.model_builder.train_horizon_model", side_effect=_fake_train_horizon_model),
                ):
                    result = run_training_job(
                        job=store.get_job(job_id),
                        store=store,
                        base_config=cfg,
                        root=tmp_path,
                        sync_models=lambda: None,
                    )

                self.assertEqual(result["sweep_variants_count"], 4)
                self.assertEqual(len(result["model_ids"]), 4)
                self.assertEqual(len(result["models"]), 4)

                log_messages = "\n".join(item["message"] for item in store.list_training_job_logs(job_id, limit=5000))
                self.assertIn("sweep resolved to 4 effective variant(s)", log_messages)
                self.assertIn("learning_rate=0.02", log_messages)
                self.assertIn("num_leaves=63", log_messages)

                for item in result["models"]:
                    model = store.get_model(str(item["model_version"]))
                    self.assertIn("variant_id", model["tags"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
