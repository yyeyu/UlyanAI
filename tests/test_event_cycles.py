from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.service.event_store import CycleCreateInput, EventCreateInput, EventStore


def _event_input(*, cycle_id: str, seq: int, total: int) -> EventCreateInput:
    now = datetime.now(timezone.utc)
    return EventCreateInput(
        asset="BTC",
        horizon="5m",
        created_at=now,
        price_t0=50000.0,
        prediction={"asset": "BTC", "horizon": "5m"},
        pred_low=49800.0,
        pred_mid=50000.0,
        pred_high=50200.0,
        model_id="btc_5m_test_model",
        model_meta={"git_commit": "test"},
        price_source="binance_spot",
        note="cycle-test",
        cycle_id=cycle_id,
        cycle_seq=seq,
        cycle_total=total,
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

