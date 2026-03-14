from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.service.event_store import EventStore


class ModelRegistryTests(unittest.TestCase):
    def test_model_row_persists_params_and_soft_delete_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventStore(Path(tmp_dir) / "events.sqlite3")
            try:
                store.upsert_model(
                    {
                        "model_id": "btc_5m_test_v1",
                        "asset": "BTC",
                        "horizon": "5m",
                        "name": "BTC 5m",
                        "created_at": "2026-03-03T00:00:00+00:00",
                        "git_commit": "abc123",
                        "train_time": "2026-03-03T00:00:00+00:00",
                        "dataset_hash": "data-hash",
                        "config_hash": "config-hash",
                        "metrics_json": {
                            "train": {"coverage_return": 0.8},
                            "val": {"coverage_return": 0.79},
                            "test": {"coverage_return": 0.78},
                            "calibrated": {"coverage_return": 0.8},
                        },
                        "params_json": {
                            "asset": "BTC",
                            "horizon": "5m",
                            "base_timeframe": "1m",
                            "steps_ahead": 5,
                            "quantiles": ["q10", "q50", "q90"],
                            "target_coverage_requested": 0.8,
                            "target_coverage_effective": 0.8,
                            "interval_mode": "symmetric",
                            "train_window_mode": "expanding",
                            "train_window_days": 365,
                            "val_days": 30,
                            "test_days": 30,
                            "walk_forward": None,
                            "feature_set_version": "feat_v1",
                            "calibration": {"method": "grid_scale"},
                            "seed": 42,
                            "data_range_start": "2025-01-01T00:00:00+00:00",
                            "data_range_end": "2026-01-01T00:00:00+00:00",
                        },
                        "is_production": False,
                        "status": "active",
                    }
                )
                active = store.get_model("btc_5m_test_v1")
                self.assertEqual(active["status"], "active")
                self.assertIsNone(active["deleted_at"])
                self.assertEqual(active["params_json"]["steps_ahead"], 5)
                self.assertIn("calibrated", active["metrics_json"])

                deleted = store.update_model_management("btc_5m_test_v1", status="deleted")
                self.assertEqual(deleted["status"], "deleted")
                self.assertIsNotNone(deleted["deleted_at"])

                restored = store.update_model_management("btc_5m_test_v1", status="active")
                self.assertEqual(restored["status"], "active")
                self.assertIsNone(restored["deleted_at"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
