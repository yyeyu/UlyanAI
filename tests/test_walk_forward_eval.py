from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.train import evaluate_walk_forward_horizon, train_horizon_model


def _build_synthetic_features_labels(rows: int = 24 * 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    trend = np.linspace(0.0, 1.0, rows)
    seasonal = np.sin(np.linspace(0.0, 25.0, rows))
    close = 50000.0 + (trend * 1000.0) + (seasonal * 120.0)
    features = pd.DataFrame(
        {
            "ts_utc": ts,
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "close": close,
            "f_trend": trend,
            "f_seasonal": seasonal,
        }
    )
    labels = pd.DataFrame(
        {
            "ts_utc": ts,
            "target_r_5m": np.clip(np.diff(np.log(close), prepend=np.log(close[0])), -0.03, 0.03),
        }
    )
    return features, labels


class WalkForwardEvalTests(unittest.TestCase):
    def test_walk_forward_uses_strictly_past_training_window(self) -> None:
        cfg = load_config("configs")
        cfg["training"]["num_boost_round"] = 3
        cfg["training"]["early_stopping_rounds"] = 0
        cfg["walk_forward"] = {
            "train_days": 20,
            "val_days": 5,
            "step_days": 5,
            "max_splits": 3,
        }
        features, labels = _build_synthetic_features_labels()

        report = evaluate_walk_forward_horizon(
            asset="BTC",
            horizon="5m",
            features_df=features,
            labels_df=labels,
            config=cfg,
        )

        folds = list(report.get("folds", []))
        self.assertGreaterEqual(len(folds), 1)
        for fold in folds:
            train_max_ts = pd.Timestamp(fold["train_max_ts"])
            val_min_ts = pd.Timestamp(fold["val_min_ts"])
            self.assertLess(train_max_ts, val_min_ts)
            self.assertIn("metrics_calibrated", fold)
        summary = dict(report.get("summary", {}))
        self.assertEqual(int(summary.get("folds_evaluated", 0)), len(folds))

    def test_train_horizon_model_saves_wf_report_and_metadata_link(self) -> None:
        cfg = load_config("configs")
        cfg["training"]["num_boost_round"] = 3
        cfg["training"]["early_stopping_rounds"] = 0
        cfg["training"]["train_window_days"] = 30
        cfg["training"]["val_days"] = 7
        cfg["training"]["test_days"] = 7
        cfg["builder_metadata"] = {
            "walk_forward": {
                "train_days": 20,
                "val_days": 5,
                "step_days": 5,
                "folds": 2,
            },
            "base_timeframe_mode": "legacy",
        }
        cfg["walk_forward"] = {
            "train_days": 20,
            "val_days": 5,
            "step_days": 5,
            "max_splits": 2,
        }
        features, labels = _build_synthetic_features_labels()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train_horizon_model(
                asset="BTC",
                horizon="5m",
                features_df=features,
                labels_df=labels,
                config=cfg,
                artifacts_root=Path(tmp_dir),
            )
            artifact_dir = Path(str(result["artifact_dir"]))
            wf_path = artifact_dir / "wf_report.json"
            self.assertTrue(wf_path.exists())

            metadata_path = artifact_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata.get("walk_forward_report_path"), "wf_report.json")
            self.assertGreaterEqual(
                int(dict(metadata.get("walk_forward_report_summary", {})).get("folds_evaluated", 0)),
                1,
            )
            params_json = dict(metadata.get("params_json", {}))
            walk_forward_meta = dict(params_json.get("walk_forward", {}))
            self.assertEqual(walk_forward_meta.get("report_path"), "wf_report.json")


if __name__ == "__main__":
    unittest.main()

