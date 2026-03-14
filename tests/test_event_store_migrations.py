from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from src.service.event_store import EventStore


def _build_legacy_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                horizon TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                git_commit TEXT NOT NULL,
                train_time TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                is_production INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                horizon TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                price_t0 REAL NOT NULL,
                pred_low REAL NOT NULL,
                pred_mid REAL NOT NULL,
                pred_high REAL NOT NULL,
                payload_json TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_meta_json TEXT NOT NULL,
                price_source TEXT NOT NULL,
                status TEXT NOT NULL,
                actual_price REAL,
                current_price REAL,
                metrics_json TEXT NOT NULL,
                note TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE cycles (
                cycle_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                horizon TEXT NOT NULL,
                total_runs INTEGER NOT NULL,
                launched_runs INTEGER NOT NULL DEFAULT 0,
                completed_runs INTEGER NOT NULL DEFAULT 0,
                cancelled_runs INTEGER NOT NULL DEFAULT 0,
                model_id TEXT,
                price_source TEXT NOT NULL,
                status TEXT NOT NULL,
                note TEXT,
                last_event_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE training_jobs (
                job_id TEXT PRIMARY KEY,
                job_kind TEXT NOT NULL,
                asset TEXT NOT NULL,
                horizons_json TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                stage TEXT NOT NULL,
                summary TEXT NOT NULL,
                params_json TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE training_job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                stage TEXT,
                message TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT INTO models (
                model_id, asset, horizon, name, created_at, git_commit, train_time,
                dataset_hash, config_hash, metrics_json, is_production
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_model",
                "BTC",
                "5m",
                "legacy_model",
                "2026-01-01T00:00:00+00:00",
                "abc123",
                "2026-01-01T00:00:00+00:00",
                "dataset",
                "config",
                "{}",
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO events (
                event_id, asset, horizon, created_at, expires_at, price_t0, pred_low, pred_mid, pred_high,
                payload_json, model_id, model_meta_json, price_source, status, actual_price, current_price,
                metrics_json, note, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_event",
                "BTC",
                "5m",
                "2026-01-01T12:07:41+00:00",
                "2026-01-01T12:12:41+00:00",
                50000.0,
                49800.0,
                50000.0,
                50200.0,
                "{\"asset\":\"BTC\",\"horizon\":\"5m\"}",
                "legacy_model",
                "{}",
                "binance_spot",
                "completed",
                50010.0,
                50010.0,
                "{\"hit\":true,\"abs_error\":10.0,\"rel_error\":0.0002,\"width_pct\":0.8,\"latency_ms\":0}",
                "legacy note",
                "2026-01-01T12:12:41+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO cycles (
                cycle_id, asset, horizon, total_runs, launched_runs, completed_runs, cancelled_runs,
                model_id, price_source, status, note, last_event_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_cycle",
                "BTC",
                "5m",
                3,
                1,
                1,
                0,
                "legacy_model",
                "binance_spot",
                "running",
                "legacy cycle",
                "legacy_event",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:05:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO training_jobs (
                job_id, job_kind, asset, horizons_json, status, progress, stage, summary, params_json, result_json,
                error, cancel_requested, created_at, started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_job",
                "train",
                "BTC",
                "[\"5m\"]",
                "succeeded",
                100,
                "done",
                "legacy summary",
                "{\"asset\":\"BTC\",\"horizons\":[\"5m\"]}",
                "{\"models\":[{\"model_version\":\"legacy_model\"}]}",
                None,
                0,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
                "2026-01-01T00:00:10+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO training_job_logs (job_id, ts, level, stage, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("legacy_job", "2026-01-01T00:00:05+00:00", "info", "train", "legacy migrated log"),
        )
        conn.commit()
    finally:
        conn.close()


def test_event_store_migrates_legacy_rows_without_data_loss() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "events.sqlite3"
        _build_legacy_db(db_path)

        store = EventStore(db_path)
        try:
            model = store.get_model("legacy_model")
            assert model["status"] == "active"
            assert model["deleted_at"] is None
            assert model["params_json"] == {}
            assert model["is_production"] is True

            event = store.get_event("legacy_event")
            assert event["note"] == "legacy note"
            assert event["params_json"] == {}
            assert event["eval_key"] == "BTC|5m|2026-01-01T12:05:00+00:00"

            cycle = store.get_cycle("legacy_cycle")
            assert cycle["mode"] == "single"
            assert cycle["models_count"] == 1
            assert cycle["model_ids"] == []
            assert cycle["align_to_boundary"] is False

            job = store.get_job("legacy_job")
            assert job["type"] == "train_model"
            assert job["status"] == "succeeded"
            assert job["summary"] == "legacy summary"
            assert job["result"]["models"][0]["model_version"] == "legacy_model"

            legacy_job = store.get_training_job("legacy_job")
            assert legacy_job["summary"] == "legacy summary"
            assert legacy_job["progress"] == 100

            logs = store.list_job_logs("legacy_job")
            assert logs
            assert logs[0]["message"] == "legacy migrated log"
        finally:
            store.close()
