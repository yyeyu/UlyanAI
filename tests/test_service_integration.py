from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import src.service.api as service_api
from src.service.event_store import CycleCreateInput, EventCreateInput, EventStore
from src.service.schemas import CreateCycleRequest, ModelBuilderValidateRequest, TrainingJobCreateRequest
from src.worker import _process_next_job


def _upsert_model(
    store: EventStore,
    model_id: str,
    *,
    horizon: str = "5m",
    is_production: bool = False,
    status: str = "active",
    base_timeframe: str | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    resolved_base_timeframe = base_timeframe or ("1m" if horizon == "5m" else horizon)
    steps_ahead = {"5m": 5, "15m": 15, "1h": 60}.get(horizon) if resolved_base_timeframe == "1m" else None
    store.upsert_model(
        {
            "model_id": model_id,
            "asset": "BTC",
            "horizon": horizon,
            "name": model_id,
            "created_at": now_iso,
            "git_commit": "test",
            "train_time": now_iso,
            "dataset_hash": "dataset",
            "config_hash": "config",
            "metrics_json": {},
            "params_json": {
                "asset": "BTC",
                "horizon": horizon,
                "base_timeframe": resolved_base_timeframe,
                "steps_ahead": steps_ahead,
                "quantiles": [0.1, 0.5, 0.9],
                "target_coverage_effective": 0.8,
            },
            "is_production": is_production,
            "status": status,
        }
    )


def _create_completed_event(
    store: EventStore,
    *,
    model_id: str,
    horizon: str,
    created_at: datetime,
    abs_error: float,
    hit: bool = True,
) -> None:
    created = store.create_event(
        EventCreateInput(
            asset="BTC",
            horizon=horizon,
            created_at=created_at,
            price_t0=50000.0,
            prediction={"asset": "BTC", "horizon": horizon},
            pred_low=49800.0,
            pred_mid=50000.0,
            pred_high=50200.0,
            model_id=model_id,
            model_meta={"git_commit": "test"},
            price_source="binance_spot",
            note="integration-test",
        )
    )
    store.complete_event(
        created["event_id"],
        actual_price=50000.0 + abs_error,
        metrics={
            "hit": hit,
            "abs_error": abs_error,
            "rel_error": abs_error / 50000.0,
            "width_pct": 0.8,
            "interval_score": abs_error + 0.5,
            "wis": abs_error + 0.25,
            "latency_ms": 0,
        },
        completed_at=datetime.now(timezone.utc),
    )


class ServiceIntegrationTests(unittest.TestCase):
    def test_train_job_result_surfaces_model_through_models_api_and_ui_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                created = store.create_job(
                    job_type="train_model",
                    params={"asset": "BTC", "horizons": ["5m"], "job_summary_hint": "integration train"},
                )

                def fake_run_training_job(*, job, store, base_config, root, sync_models):
                    _upsert_model(store, "builder_model_5m")
                    return {
                        "warnings": [],
                        "model_ids": ["builder_model_5m"],
                        "models": [{"model_version": "builder_model_5m", "horizon": "5m"}],
                    }

                with (
                    patch("src.worker.run_training_job", side_effect=fake_run_training_job),
                    patch("src.worker.sync_models_registry", return_value=None),
                ):
                    processed = _process_next_job(
                        store=store,
                        config={},
                        root=tmp_path,
                        artifacts_root=tmp_path,
                    )
                self.assertTrue(processed)
                job = store.get_job(str(created["job_id"]))
                self.assertEqual(job["status"], "succeeded")
                self.assertIn("builder_model_5m", job["result"]["model_ids"])

                with (
                    patch.object(service_api, "EVENT_STORE", store),
                    patch.object(service_api, "_sync_models_registry", return_value=None),
                ):
                    models_response = service_api.list_models(
                        asset="BTC",
                        horizon="5m",
                        include_archived=False,
                        include_deleted=False,
                        limit=20,
                    )
                self.assertTrue(any(item.model_id == "builder_model_5m" for item in models_response.items))

                html = Path("web/index.html").read_text(encoding="utf-8").lower()
                self.assertIn("buildermodelsbody", html)
                self.assertIn("builderaddjobmodelstocomparebtn", html)
            finally:
                store.close()

    def test_model_management_constraints_apply_through_service_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                _upsert_model(store, "managed_model", is_production=False)
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
                        model_id="managed_model",
                        model_meta={"git_commit": "test"},
                        price_source="binance_spot",
                        note="model-ref",
                    )
                )
                self.assertTrue(created["event_id"])

                with (
                    patch.object(service_api, "EVENT_STORE", store),
                    patch.object(service_api, "_sync_models_registry", return_value=None),
                    patch.object(service_api, "_artifact_dir_for_model", return_value=tmp_path / "artifacts"),
                ):
                    archived = service_api.archive_model("managed_model")
                    self.assertEqual(archived.status, "archived")

                    with self.assertRaises(HTTPException) as promote_exc:
                        service_api.promote_model("managed_model")
                    self.assertEqual(promote_exc.exception.status_code, 400)

                    restored = service_api.unarchive_model("managed_model")
                    self.assertEqual(restored.status, "active")

                    deleted = service_api.soft_delete_model("managed_model")
                    self.assertEqual(deleted.status, "deleted")

                    visible_default = service_api.list_models(
                        asset="BTC",
                        horizon="5m",
                        include_archived=False,
                        include_deleted=False,
                        limit=20,
                    )
                    self.assertFalse(any(item.model_id == "managed_model" for item in visible_default.items))

                    visible_deleted = service_api.list_models(
                        asset="BTC",
                        horizon="5m",
                        include_archived=False,
                        include_deleted=True,
                        limit=20,
                    )
                    self.assertTrue(any(item.model_id == "managed_model" for item in visible_deleted.items))

                    with self.assertRaises(HTTPException) as purge_exc:
                        service_api.purge_model("managed_model", confirm=True)
                    self.assertEqual(purge_exc.exception.status_code, 400)
            finally:
                store.close()

    def test_compare_api_and_tournament_cycle_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                _upsert_model(store, "model_a", is_production=True, base_timeframe="1m")
                _upsert_model(store, "model_b", base_timeframe="1m")
                _upsert_model(store, "model_c", horizon="15m", base_timeframe="15m")

                class DummyBundle:
                    metadata = {"calibrator": {"target_coverage": 0.8}}

                class DummyPrediction:
                    def __init__(self) -> None:
                        self.primary = service_api.ReturnQuantiles(q10=-0.01, q50=0.0, q90=0.01)
                        self.extra_quantiles = {}

                with (
                    patch.object(service_api, "EVENT_STORE", store),
                    patch.object(service_api, "_load_bundle_by_id_cached", return_value=DummyBundle()),
                    patch.object(
                        service_api.CANDLE_CACHE,
                        "latest_feature_row",
                        return_value=({"close": 50000.0}, 50000.0, datetime(2026, 1, 1, tzinfo=timezone.utc), False),
                    ),
                    patch.object(
                        service_api.CANDLE_CACHE,
                        "latest_spot_quote",
                        return_value=(50000.0, datetime(2026, 1, 1, tzinfo=timezone.utc), False),
                    ),
                    patch.object(service_api, "predict_bundle_outputs", return_value=DummyPrediction()),
                ):
                    cycle = service_api.create_cycle(
                        CreateCycleRequest(
                            asset="BTC",
                            horizon="5m",
                            runs=1,
                            model_id=None,
                            model_ids=["model_a", "model_b"],
                            price_source="binance_spot",
                            mode="tournament",
                            note="integration-tournament",
                        )
                    )

                    self.assertEqual(cycle.mode, "tournament")
                    self.assertEqual(cycle.launched_runs, 1)

                    payload = store.list_events(cycle_id=cycle.cycle_id, page=1, page_size=20)
                    self.assertEqual(payload["total"], 2)
                    self.assertEqual(len({item["created_at"] for item in payload["items"]}), 1)
                    self.assertEqual(len({item["eval_key"] for item in payload["items"]}), 1)

                    for index, item in enumerate(payload["items"], start=1):
                        store.complete_event(
                            item["event_id"],
                            actual_price=50000.0 + index,
                            metrics={
                                "hit": True,
                                "abs_error": float(index),
                                "rel_error": 0.001,
                                "width_pct": 0.8,
                                "interval_score": float(index) + 0.5,
                                "wis": float(index) + 0.25,
                                "latency_ms": 0,
                            },
                            completed_at=datetime.now(timezone.utc),
                        )

                    completed_cycle = store.get_cycle(cycle.cycle_id)
                    self.assertEqual(completed_cycle["completed_runs"], 1)
                    self.assertEqual(completed_cycle["status"], "completed")

                    created_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                    _create_completed_event(
                        store,
                        model_id="model_c",
                        horizon="15m",
                        created_at=created_at,
                        abs_error=5.0,
                        hit=True,
                    )

                    simple = service_api.compare_scoreboard(
                        days=30,
                        asset="BTC",
                        horizon="5m",
                        cycle_id=cycle.cycle_id,
                        mode="simple",
                        model_ids="model_a,model_b",
                        exclude_stale=False,
                        baseline_model_id=None,
                        columns="coverage,avg_abs_error,base_timeframe,steps_ahead,quantiles_count,target_coverage_effective,avg_interval_score,avg_wis",
                    )
                    self.assertEqual(simple.mode, "simple")
                    self.assertEqual(simple.total_completed, 2)
                    model_a_row = next(item for item in simple.items if item.model_id == "model_a")
                    self.assertEqual(model_a_row.base_timeframe, "1m")
                    self.assertEqual(model_a_row.steps_ahead, 5)
                    self.assertEqual(model_a_row.quantiles_count, 3)
                    self.assertAlmostEqual(model_a_row.target_coverage_effective or 0.0, 0.8)
                    self.assertIsNotNone(model_a_row.avg_interval_score)
                    self.assertIsNotNone(model_a_row.avg_wis)

                    matched = service_api.compare_scoreboard(
                        days=30,
                        asset="BTC",
                        horizon="5m",
                        cycle_id=cycle.cycle_id,
                        mode="matched",
                        model_ids="model_a,model_b",
                        exclude_stale=False,
                        baseline_model_id="model_a",
                        columns="coverage,avg_abs_error,avg_interval_score,avg_wis,group_wins",
                    )
                    self.assertEqual(matched.mode, "matched")
                    self.assertEqual(matched.total_groups, 1)
                    model_b_row = next(item for item in matched.items if item.model_id == "model_b")
                    self.assertIsNotNone(model_b_row.delta_avg_interval_score)
                    self.assertIsNotNone(model_b_row.delta_avg_wis)

                    grouped = service_api.compare_scoreboard(
                        days=30,
                        asset="BTC",
                        horizon=None,
                        cycle_id=None,
                        mode="grouped",
                        model_ids="model_a,model_b,model_c",
                        exclude_stale=False,
                        baseline_model_id=None,
                        columns="asset,horizon,group_key,avg_abs_error",
                    )
                    self.assertEqual(grouped.mode, "grouped")
                    self.assertTrue(any(item.group_key == "BTC|15m" for item in grouped.items))

                    series = service_api.compare_series(
                        days=30,
                        asset="BTC",
                        horizon="5m",
                        cycle_id=cycle.cycle_id,
                        mode="matched",
                        model_ids="model_a,model_b",
                        exclude_stale=False,
                        metric="interval_score",
                        rolling_days=3,
                    )
                    self.assertTrue(series.series)

                    hist = service_api.compare_hist(
                        days=30,
                        asset="BTC",
                        horizon=None,
                        cycle_id=None,
                        mode="grouped",
                        model_ids="model_a,model_b,model_c",
                        exclude_stale=False,
                        metric="wis",
                        bins=4,
                    )
                    self.assertTrue(hist.series)
            finally:
                store.close()

    def test_cycle_summary_endpoint_and_orchestrator_apply_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                _upsert_model(store, "model_a", is_production=True, base_timeframe="1m")
                _upsert_model(store, "model_b", base_timeframe="1m")
                cycle = store.create_cycle(
                    CycleCreateInput(
                        asset="BTC",
                        horizon="5m",
                        total_runs=6,
                        model_id=None,
                        price_source="binance_spot",
                        mode="tournament",
                        model_ids=("model_a", "model_b"),
                        models_count=2,
                        note="summary-orchestrator",
                    )
                )
                cycle_id = str(cycle["cycle_id"])

                run_errors = [
                    {"model_a": 5.0, "model_b": 20.0},
                    {"model_a": 4.0, "model_b": 22.0},
                    {"model_a": 6.0, "model_b": 24.0},
                ]
                for seq, errors in enumerate(run_errors, start=1):
                    created_at = datetime.now(timezone.utc)
                    batch = store.create_events_batch_for_cycle(
                        cycle_id,
                        seq,
                        6,
                        [
                            EventCreateInput(
                                asset="BTC",
                                horizon="5m",
                                created_at=created_at,
                                price_t0=50000.0,
                                prediction={"asset": "BTC", "horizon": "5m"},
                                pred_low=49800.0,
                                pred_mid=50000.0,
                                pred_high=50200.0,
                                model_id="model_a",
                                model_meta={"git_commit": "test"},
                                price_source="binance_spot",
                                cycle_id=cycle_id,
                                cycle_seq=seq,
                                cycle_total=6,
                            ),
                            EventCreateInput(
                                asset="BTC",
                                horizon="5m",
                                created_at=created_at,
                                price_t0=50000.0,
                                prediction={"asset": "BTC", "horizon": "5m"},
                                pred_low=49800.0,
                                pred_mid=50000.0,
                                pred_high=50200.0,
                                model_id="model_b",
                                model_meta={"git_commit": "test"},
                                price_source="binance_spot",
                                cycle_id=cycle_id,
                                cycle_seq=seq,
                                cycle_total=6,
                            ),
                        ],
                    )
                    for event in batch:
                        model_id = str(event["model_id"])
                        abs_error = float(errors[model_id])
                        store.complete_event(
                            event["event_id"],
                            actual_price=50000.0 + abs_error,
                            metrics={
                                "hit": True,
                                "abs_error": abs_error,
                                "rel_error": abs_error / 50000.0,
                                "width_pct": 0.7 if model_id == "model_a" else 1.0,
                                "interval_score": abs_error + 0.3,
                                "wis": abs_error + 0.2,
                                "latency_ms": 0,
                            },
                            completed_at=datetime.now(timezone.utc),
                        )

                with (
                    patch.object(service_api, "EVENT_STORE", store),
                    patch.object(
                        service_api,
                        "CONFIG",
                        {
                            "service": {
                                "tournament_orchestrator": {
                                    "enabled": True,
                                    "min_paired_runs": 3,
                                    "win_rate_threshold": 0.4,
                                    "delta_width_min_runs": 2,
                                    "delta_width_ci_max": 1.0,
                                }
                            }
                        },
                    ),
                ):
                    summary = service_api.get_cycle_summary(cycle_id=cycle_id)
                    self.assertEqual(summary.recommendation, "winner")
                    self.assertEqual(summary.winner_model_id, "model_a")
                    self.assertGreaterEqual(summary.paired_runs, 3)
                    self.assertTrue(summary.leaderboard)
                    self.assertEqual(summary.leaderboard[0].model_id, "model_a")

                    service_api._run_tournament_orchestrator(limit=20)

                cycle_after = store.get_cycle(cycle_id)
                self.assertEqual(cycle_after["total_runs"], 3)
                self.assertEqual(cycle_after["status"], "completed")
            finally:
                store.close()

    def test_wf_eval_job_can_be_created_via_lab_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = EventStore(tmp_path / "events.sqlite3")
            try:
                payload = TrainingJobCreateRequest(
                    asset="BTC",
                    horizons=["5m"],
                    walk_forward_enabled=True,
                    wf_train_days=20,
                    wf_val_days=5,
                    wf_step_days=5,
                    wf_folds=2,
                )
                with patch.object(service_api, "EVENT_STORE", store):
                    created = service_api.create_wf_eval_jobs(payload)
                self.assertEqual(len(created.items), 1)
                self.assertEqual(created.items[0].job_kind, "wf_eval")
                self.assertEqual(created.items[0].status, "pending")
            finally:
                store.close()

    def test_lab_validate_returns_generic_sweep_preview(self) -> None:
        payload = ModelBuilderValidateRequest(
            asset="BTC",
            horizons=["5m"],
            sweep_axes=[
                {"path": "target_coverage_percent", "mode": "list", "values": [80, 81, 82]},
                {"path": "feature_groups.rsi", "mode": "list", "values": [True, False]},
            ],
        )

        response = service_api.validate_model_builder(payload, job_type="sweep_train")

        self.assertTrue(response.ok)
        self.assertEqual(response.sweep_preview.requested_total, 6)
        self.assertEqual(response.sweep_preview.effective_total, 4)
        self.assertEqual(response.sweep_preview.duplicate_count, 2)
        self.assertTrue(response.sweep_preview.axes)


if __name__ == "__main__":
    unittest.main()
