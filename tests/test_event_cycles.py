from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.service.event_store import CycleCreateInput, EventCreateInput, EventStore


def _event_input(
    *,
    cycle_id: str,
    seq: int,
    total: int,
    model_id: str = "btc_5m_test_model",
    created_at: datetime | None = None,
) -> EventCreateInput:
    now = datetime.now(timezone.utc)
    return EventCreateInput(
        asset="BTC",
        horizon="5m",
        created_at=created_at or now,
        price_t0=50000.0,
        prediction={"asset": "BTC", "horizon": "5m"},
        pred_low=49800.0,
        pred_mid=50000.0,
        pred_high=50200.0,
        model_id=model_id,
        model_meta={"git_commit": "test"},
        price_source="binance_spot",
        note="cycle-test",
        cycle_id=cycle_id,
        cycle_seq=seq,
        cycle_total=total,
    )


def _upsert_model(store: EventStore, model_id: str, *, is_production: bool = False) -> None:
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
            "metrics_json": {},
            "params_json": {"asset": "BTC", "horizon": "5m", "base_timeframe": "5m"},
            "is_production": is_production,
            "status": "active",
        }
    )


def test_cycle_progress_and_sequential_guard(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=2,
            model_id=None,
            price_source="binance_spot",
            note="smoke-cycle",
        )
    )
    cycle_id = str(cycle["cycle_id"])

    first = store.create_event(_event_input(cycle_id=cycle_id, seq=1, total=2))
    cycle_after_first = store.get_cycle(cycle_id)
    assert cycle_after_first["launched_runs"] == 1
    assert cycle_after_first["status"] == "running"

    with pytest.raises(ValueError, match="active event"):
        store.create_event(_event_input(cycle_id=cycle_id, seq=2, total=2))

    store.complete_event(
        first["event_id"],
        actual_price=50100.0,
        metrics={"hit": True, "abs_error": 100.0, "rel_error": 0.002, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    cycle_after_complete = store.get_cycle(cycle_id)
    assert cycle_after_complete["completed_runs"] == 1
    assert cycle_after_complete["status"] == "running"

    second = store.create_event(_event_input(cycle_id=cycle_id, seq=2, total=2))
    cycle_after_second = store.get_cycle(cycle_id)
    assert cycle_after_second["launched_runs"] == 2

    store.cancel_event(second["event_id"], note="manual")
    cycle_done = store.get_cycle(cycle_id)
    assert cycle_done["cancelled_runs"] == 1
    assert cycle_done["status"] == "completed"

    filtered = store.list_events(cycle_id=cycle_id, page=1, page_size=100)
    assert filtered["total"] == 2
    assert all(item["cycle_id"] == cycle_id for item in filtered["items"])
    assert all(item["eval_key"].startswith("BTC|5m|") for item in filtered["items"])
    assert all(item["params_json"]["model_id"] == "btc_5m_test_model" for item in filtered["items"])


def test_backfill_event_eval_keys(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    created = store.create_event(_event_input(cycle_id="", seq=1, total=1))
    event_id = str(created["event_id"])
    with store._lock:
        store._exec("UPDATE events SET eval_key='' WHERE event_id=?", (event_id,))
        store._conn.commit()

    updated = store.backfill_event_eval_keys(only_missing=True)
    assert updated == 1

    refreshed = store.get_event(event_id)
    assert refreshed["eval_key"].startswith("BTC|5m|")


def test_eval_key_uses_exact_bucket_floor(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    created = store.create_event(
        _event_input(
            cycle_id="",
            seq=1,
            total=1,
            created_at=datetime(2026, 1, 1, 12, 7, 41, tzinfo=timezone.utc),
        )
    )

    assert created["eval_key"] == "BTC|5m|2026-01-01T12:05:00+00:00"


def test_tournament_cycle_counts_runs_not_events(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=2,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
            note="tournament-smoke",
        )
    )
    cycle_id = str(cycle["cycle_id"])
    created_at = datetime.now(timezone.utc)

    batch_1 = store.create_events_batch_for_cycle(
        cycle_id,
        1,
        2,
        [
            _event_input(cycle_id=cycle_id, seq=1, total=2, model_id="model_a", created_at=created_at),
            _event_input(cycle_id=cycle_id, seq=1, total=2, model_id="model_b", created_at=created_at),
        ],
    )
    assert len(batch_1) == 2
    assert len({item["event_id"] for item in batch_1}) == 2
    assert len({item["created_at"] for item in batch_1}) == 1
    assert len({item["eval_key"] for item in batch_1}) == 1
    after_launch = store.get_cycle(cycle_id)
    assert after_launch["mode"] == "tournament"
    assert after_launch["models_count"] == 2
    assert after_launch["launched_runs"] == 1
    assert after_launch["completed_runs"] == 0
    assert after_launch["cancelled_runs"] == 0

    store.complete_event(
        batch_1[0]["event_id"],
        actual_price=50100.0,
        metrics={"hit": True, "abs_error": 100.0, "rel_error": 0.002, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    still_waiting = store.get_cycle(cycle_id)
    assert still_waiting["completed_runs"] == 0

    store.complete_event(
        batch_1[1]["event_id"],
        actual_price=50090.0,
        metrics={"hit": True, "abs_error": 90.0, "rel_error": 0.0018, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    after_complete = store.get_cycle(cycle_id)
    assert after_complete["completed_runs"] == 1
    assert after_complete["cancelled_runs"] == 0
    assert after_complete["status"] == "running"

    created_at_2 = datetime.now(timezone.utc)
    batch_2 = store.create_events_batch_for_cycle(
        cycle_id,
        2,
        2,
        [
            _event_input(cycle_id=cycle_id, seq=2, total=2, model_id="model_a", created_at=created_at_2),
            _event_input(cycle_id=cycle_id, seq=2, total=2, model_id="model_b", created_at=created_at_2),
        ],
    )
    store.cancel_event(batch_2[0]["event_id"], note="manual")
    cycle_mid = store.get_cycle(cycle_id)
    assert cycle_mid["cancelled_runs"] == 0
    store.complete_event(
        batch_2[1]["event_id"],
        actual_price=50080.0,
        metrics={"hit": False, "abs_error": 80.0, "rel_error": 0.0016, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    cycle_done = store.get_cycle(cycle_id)
    assert cycle_done["launched_runs"] == 2
    assert cycle_done["completed_runs"] == 1
    assert cycle_done["cancelled_runs"] == 1
    assert cycle_done["status"] == "completed"


def test_tournament_batch_insert_is_transactional(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=1,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
        )
    )
    cycle_id = str(cycle["cycle_id"])
    created_at = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="same created_at"):
        store.create_events_batch_for_cycle(
            cycle_id,
            1,
            1,
            [
                _event_input(cycle_id=cycle_id, seq=1, total=1, model_id="model_a", created_at=created_at),
                _event_input(
                    cycle_id=cycle_id,
                    seq=1,
                    total=1,
                    model_id="model_b",
                    created_at=created_at.replace(second=(created_at.second + 1) % 60),
                ),
            ],
        )

    assert store.list_events(cycle_id=cycle_id, page=1, page_size=100)["total"] == 0
    cycle_after = store.get_cycle(cycle_id)
    assert cycle_after["launched_runs"] == 0


def test_compare_metrics_matched_uses_strict_intersection(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    _upsert_model(store, "model_a", is_production=True)
    _upsert_model(store, "model_b")

    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=1,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
        )
    )
    cycle_id = str(cycle["cycle_id"])
    created_at = datetime.now(timezone.utc).replace(second=41, microsecond=0)
    batch = store.create_events_batch_for_cycle(
        cycle_id,
        1,
        1,
        [
            _event_input(cycle_id=cycle_id, seq=1, total=1, model_id="model_a", created_at=created_at),
            _event_input(cycle_id=cycle_id, seq=1, total=1, model_id="model_b", created_at=created_at),
        ],
    )

    for index, event in enumerate(batch, start=1):
        store.complete_event(
            event["event_id"],
            actual_price=50000.0 + index,
            metrics={"hit": True, "abs_error": float(index), "rel_error": 0.001, "width_pct": 0.8, "latency_ms": 0},
            completed_at=datetime.now(timezone.utc),
        )

    extra = store.create_event(
        _event_input(
            cycle_id="",
            seq=1,
            total=1,
            model_id="model_a",
            created_at=created_at + timedelta(minutes=6),
        )
    )
    store.complete_event(
        extra["event_id"],
        actual_price=50010.0,
        metrics={"hit": False, "abs_error": 10.0, "rel_error": 0.002, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )

    compare = store.compare_metrics(
        days=30,
        asset="BTC",
        horizon="5m",
        model_ids=["model_a", "model_b"],
        mode="matched",
    )

    assert compare["mode"] == "matched"
    assert compare["total_groups"] == 1
    assert compare["total_events"] == 2
    assert [item["total_events"] for item in compare["items"]] == [1, 1]


def test_compare_metrics_matched_groups_rank_models_fairly(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    _upsert_model(store, "model_a", is_production=True)
    _upsert_model(store, "model_b")

    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=1,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
        )
    )
    cycle_id = str(cycle["cycle_id"])
    created_at = datetime.now(timezone.utc)
    batch = store.create_events_batch_for_cycle(
        cycle_id,
        1,
        1,
        [
            _event_input(cycle_id=cycle_id, seq=1, total=1, model_id="model_a", created_at=created_at),
            _event_input(cycle_id=cycle_id, seq=1, total=1, model_id="model_b", created_at=created_at),
        ],
    )

    store.complete_event(
        batch[0]["event_id"],
        actual_price=50120.0,
        metrics={"hit": False, "abs_error": 120.0, "rel_error": 0.0024, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    store.complete_event(
        batch[1]["event_id"],
        actual_price=50040.0,
        metrics={"hit": True, "abs_error": 40.0, "rel_error": 0.0008, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )

    compare = store.compare_metrics(
        days=30,
        asset="BTC",
        horizon="5m",
        cycle_id=cycle_id,
        model_ids=["model_a", "model_b"],
        mode="matched",
    )

    assert compare["mode"] == "matched"
    assert compare["total_groups"] == 1
    assert compare["total_events"] == 2
    assert compare["groups"][0]["best_model_id"] == "model_b"
    assert compare["groups"][0]["event_count"] == 2
    assert compare["items"][0]["model_id"] == "model_b"
    assert compare["items"][0]["group_wins"] == 1
    assert compare["items"][0]["matched_groups"] == 1

    scoreboard = store.compare_scoreboard(
        days=30,
        asset="BTC",
        horizon="5m",
        cycle_id=cycle_id,
        model_ids=["model_a", "model_b"],
        mode="matched",
        exclude_stale=True,
        baseline_model_id="model_a",
        columns=["coverage", "avg_abs_error", "group_wins"],
    )
    assert scoreboard["mode"] == "matched"
    assert scoreboard["total_groups"] == 1
    assert scoreboard["items"][0]["model_id"] == "model_b"
    assert scoreboard["items"][0]["delta_avg_abs_error"] is not None

    series = store.compare_series(
        days=30,
        asset="BTC",
        horizon="5m",
        cycle_id=cycle_id,
        model_ids=["model_a", "model_b"],
        mode="matched",
        metric="coverage",
        rolling_days=3,
    )
    assert series["series"]
    assert series["series"][0]["points"]

    hist = store.compare_hist(
        days=30,
        asset="BTC",
        horizon="5m",
        cycle_id=cycle_id,
        model_ids=["model_a", "model_b"],
        mode="matched",
        metric="abs_error",
        bins=4,
    )
    assert hist["series"]
    assert len(hist["series"][0]["bins"]) == 4


def test_tournament_cycle_summary_recommends_winner_and_supports_early_stop(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    _upsert_model(store, "model_a", is_production=True)
    _upsert_model(store, "model_b")

    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=5,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
        )
    )
    cycle_id = str(cycle["cycle_id"])

    run_errors = [
        {"model_a": 5.0, "model_b": 25.0},
        {"model_a": 6.0, "model_b": 30.0},
        {"model_a": 4.0, "model_b": 27.0},
    ]
    for seq, errors in enumerate(run_errors, start=1):
        created_at = datetime.now(timezone.utc) + timedelta(minutes=seq * 6)
        batch = store.create_events_batch_for_cycle(
            cycle_id,
            seq,
            5,
            [
                _event_input(cycle_id=cycle_id, seq=seq, total=5, model_id="model_a", created_at=created_at),
                _event_input(cycle_id=cycle_id, seq=seq, total=5, model_id="model_b", created_at=created_at),
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
                    "width_pct": 0.7 if model_id == "model_a" else 1.1,
                    "latency_ms": 0,
                },
                completed_at=datetime.now(timezone.utc),
            )

    summary = store.cycle_summary(
        cycle_id,
        orchestrator_rules={
            "min_paired_runs": 3,
            "win_rate_threshold": 0.40,
            "delta_width_min_runs": 2,
            "delta_width_ci_max": 1.0,
        },
    )
    assert summary["paired_runs"] == 3
    assert summary["recommendation"] == "winner"
    assert summary["winner_model_id"] == "model_a"
    assert summary["early_stop_triggered"] is True
    assert summary["leaderboard"][0]["model_id"] == "model_a"
    assert len(summary["paired_metrics"]) == 3

    changed = store.apply_tournament_early_stop(cycle_id)
    assert changed is True
    cycle_after = store.get_cycle(cycle_id)
    assert cycle_after["total_runs"] == 3
    assert cycle_after["status"] == "completed"


def test_tournament_cycle_summary_recommendation_continue_without_confident_signal(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    _upsert_model(store, "model_a", is_production=True)
    _upsert_model(store, "model_b")

    cycle = store.create_cycle(
        CycleCreateInput(
            asset="BTC",
            horizon="5m",
            total_runs=3,
            model_id=None,
            price_source="binance_spot",
            mode="tournament",
            model_ids=("model_a", "model_b"),
            models_count=2,
        )
    )
    cycle_id = str(cycle["cycle_id"])
    created_at = datetime.now(timezone.utc)
    batch = store.create_events_batch_for_cycle(
        cycle_id,
        1,
        3,
        [
            _event_input(cycle_id=cycle_id, seq=1, total=3, model_id="model_a", created_at=created_at),
            _event_input(cycle_id=cycle_id, seq=1, total=3, model_id="model_b", created_at=created_at),
        ],
    )
    store.complete_event(
        batch[0]["event_id"],
        actual_price=50010.0,
        metrics={"hit": True, "abs_error": 10.0, "rel_error": 0.0002, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )
    store.complete_event(
        batch[1]["event_id"],
        actual_price=50020.0,
        metrics={"hit": True, "abs_error": 20.0, "rel_error": 0.0004, "width_pct": 0.8, "latency_ms": 0},
        completed_at=datetime.now(timezone.utc),
    )

    summary = store.cycle_summary(cycle_id, orchestrator_rules={"min_paired_runs": 5, "win_rate_threshold": 0.8})
    assert summary["paired_runs"] == 1
    assert summary["recommendation"] == "continue"
    assert summary["early_stop_triggered"] is False
