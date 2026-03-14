from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import src.service.api as service_api
from src.config import RuntimePaths
from src.reporting.report_builder import generate_experiment_report_artifacts
from src.service.event_store import EventCreateInput, EventStore
from src.service.schemas import GenerateReportRequest


def _upsert_model(store: EventStore, model_id: str, *, experiment_id: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    store.upsert_model(
        {
            "model_id": model_id,
            "asset": "BTC",
            "horizon": "5m",
            "name": model_id,
            "created_at": now_iso,
            "git_commit": "test",
            "train_time": now_iso,
            "dataset_hash": "dataset",
            "config_hash": "config",
            "metrics_json": {"coverage_val": 0.81, "avg_width_pct": 0.67, "avg_abs_error": 11.3},
            "params_json": {
                "base_timeframe": "1m",
                "steps_ahead": 5,
                "quantiles": [0.1, 0.5, 0.9],
                "calibration_method": "grid_scale",
                "scale_selection_rule": "min_abs_coverage_gap",
                "scale_grid": [0.8, 1.0, 1.2],
                "hyperparams": {"learning_rate": 0.05, "num_leaves": 31},
                "calibrator": {"chosen_scale": 1.0},
            },
            "is_production": True,
            "status": "active",
            "experiment_id": experiment_id,
        }
    )


def _create_completed_event(store: EventStore, *, model_id: str) -> None:
    created = store.create_event(
        EventCreateInput(
            asset="BTC",
            horizon="5m",
            created_at=datetime.now(timezone.utc),
            price_t0=50000.0,
            prediction={"asset": "BTC", "horizon": "5m"},
            pred_low=49800.0,
            pred_mid=50000.0,
            pred_high=50200.0,
            model_id=model_id,
            model_meta={"git_commit": "test"},
            price_source="binance_spot",
            note="report-test",
        )
    )
    store.complete_event(
        created["event_id"],
        actual_price=50005.0,
        metrics={
            "hit": True,
            "abs_error": 5.0,
            "rel_error": 5.0 / 50000.0,
            "width_pct": 0.8,
            "interval_score": 5.25,
            "wis": 5.1,
            "latency_ms": 0,
        },
        completed_at=datetime.now(timezone.utc),
    )


class ReportBuilderTests(unittest.TestCase):
    def test_generate_experiment_report_creates_txt_and_pdf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifacts_root = tmp_path / "artifacts"
            store = EventStore(artifacts_root / "db" / "events.sqlite3")
            try:
                _upsert_model(store, "report_model_1", experiment_id="exp_report_1")
                _create_completed_event(store, model_id="report_model_1")

                def fake_pdf_writer(path: Path, content: str) -> None:
                    path.write_text(content, encoding="utf-8")

                with patch("src.reporting.report_builder._write_pdf_from_text", side_effect=fake_pdf_writer):
                    payload = generate_experiment_report_artifacts(
                        store=store,
                        artifacts_root=artifacts_root,
                        experiment_id="exp_report_1",
                    )

                txt_path = Path(payload["report_txt_path"])
                pdf_path = Path(payload["report_pdf_path"])
                self.assertTrue(txt_path.exists())
                self.assertTrue(pdf_path.exists())
                txt_content = txt_path.read_text(encoding="utf-8")
                self.assertEqual(txt_content, pdf_path.read_text(encoding="utf-8"))
                self.assertIn("1. Executive Summary", txt_content)
                self.assertIn("10. Recommendation and Appendix", txt_content)
                self.assertIn("report_model_1", txt_content)
            finally:
                store.close()

    def test_generate_report_api_and_download_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifacts_root = tmp_path / "artifacts"
            data_root = tmp_path / "data"
            artifacts_root.mkdir(parents=True, exist_ok=True)
            data_root.mkdir(parents=True, exist_ok=True)
            store = EventStore(artifacts_root / "db" / "events.sqlite3")
            try:
                _upsert_model(store, "report_model_api", experiment_id="exp_report_api")
                _create_completed_event(store, model_id="report_model_api")

                def fake_pdf_writer(path: Path, content: str) -> None:
                    path.write_text(content, encoding="utf-8")

                with (
                    patch("src.reporting.report_builder._write_pdf_from_text", side_effect=fake_pdf_writer),
                    patch.object(service_api, "EVENT_STORE", store),
                    patch.object(
                        service_api,
                        "PATHS",
                        RuntimePaths(root=tmp_path, data_root=data_root, artifacts_root=artifacts_root),
                    ),
                ):
                    response = service_api.generate_report(GenerateReportRequest(experiment_id="exp_report_api"))
                    self.assertEqual(response.experiment_id, "exp_report_api")
                    self.assertTrue(Path(response.report_txt_path).exists())
                    self.assertTrue(Path(response.report_pdf_path).exists())
                    self.assertIn("/api/reports/exp_report_api/download?format=pdf", response.report_pdf_url)

                    txt_download = service_api.download_report("exp_report_api", format="txt")
                    self.assertEqual(txt_download.media_type, "text/plain; charset=utf-8")
                    self.assertTrue(str(txt_download.path).endswith("report.txt"))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

