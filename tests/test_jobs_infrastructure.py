from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.service.event_store import EventStore
from src.worker import _process_next_job


class JobsInfrastructureTests(unittest.TestCase):
    def test_generic_jobs_table_and_legacy_wrappers_share_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventStore(Path(tmp_dir) / "events.sqlite3")
            try:
                created = store.create_job(
                    job_type="train_model",
                    params={"asset": "BTC", "horizons": ["5m"], "job_summary_hint": "train smoke"},
                )
                self.assertEqual(created["status"], "pending")
                self.assertEqual(created["stage"], "queued")
                self.assertAlmostEqual(created["progress"], 0.0, places=8)
                self.assertIn("job created", created["logs_text"])

                legacy = store.get_training_job(str(created["job_id"]))
                self.assertEqual(legacy["job_kind"], "train")
                self.assertEqual(legacy["summary"], "train smoke")
                self.assertEqual(legacy["progress"], 0)

                claimed = store.claim_next_job(job_types=["train_model"])
                self.assertIsNotNone(claimed)
                assert claimed is not None
                self.assertEqual(claimed["status"], "running")
                self.assertEqual(claimed["stage"], "data")
                self.assertAlmostEqual(claimed["progress"], 0.01, places=8)

                running = store.request_cancel_job(str(created["job_id"]))
                self.assertEqual(running["status"], "running")
                self.assertTrue(running["cancel_requested"])

                finished = store.update_job(
                    str(created["job_id"]),
                    status="succeeded",
                    progress=1.0,
                    result={"models": [{"model_version": "m1"}]},
                    error_text=None,
                    finished=True,
                )
                self.assertEqual(finished["status"], "succeeded")
                self.assertEqual(finished["stage"], "done")

                legacy_finished = store.get_training_job(str(created["job_id"]))
                self.assertEqual(legacy_finished["status"], "succeeded")
                self.assertEqual(legacy_finished["stage"], "done")
                self.assertEqual(legacy_finished["progress"], 100)
            finally:
                store.close()

    def test_pending_cancel_marks_job_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EventStore(Path(tmp_dir) / "events.sqlite3")
            try:
                created = store.create_job(
                    job_type="sweep_train",
                    params={"asset": "BTC", "horizons": ["5m", "15m"]},
                )
                canceled = store.request_cancel_job(str(created["job_id"]))
                self.assertEqual(canceled["status"], "canceled")
                self.assertEqual(canceled["stage"], "done")
                self.assertTrue(canceled["cancel_requested"])
            finally:
                store.close()

    def test_worker_processes_wf_eval_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                created = store.create_job(
                    job_type="wf_eval",
                    params={"asset": "BTC", "horizons": ["5m"]},
                )
                with patch(
                    "src.worker.run_walk_forward_job",
                    return_value={
                        "reports": [
                            {
                                "horizon": "5m",
                                "summary": {"folds_evaluated": 2},
                            }
                        ],
                        "folds_total": 2,
                    },
                ):
                    processed = _process_next_job(
                        store=store,
                        config={},
                        root=tmp_path,
                        artifacts_root=tmp_path,
                    )
                self.assertTrue(processed)
                succeeded = store.get_job(str(created["job_id"]))
                self.assertEqual(succeeded["status"], "succeeded")
                self.assertEqual(succeeded["stage"], "done")
                self.assertEqual(int(succeeded["result"].get("folds_total", 0)), 2)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
