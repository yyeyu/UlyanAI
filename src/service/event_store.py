"""SQLite-backed storage for Web GUI events, models, and alerts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

HORIZON_TO_SECONDS: dict[str, int] = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _parse_utc(ts: str) -> datetime:
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_load(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    try:
        return json.loads(payload)
    except Exception:
        return default


@dataclass(frozen=True)
class EventCreateInput:
    asset: str
    horizon: str
    created_at: datetime
    price_t0: float
    prediction: dict[str, Any]
    pred_low: float
    pred_mid: float
    pred_high: float
    model_id: str
    model_meta: dict[str, Any]
    price_source: str
    note: str | None = None


class EventStore:
    """Thread-safe SQLite storage used by the dashboard backend."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def _init_schema(self) -> None:
        with self._lock:
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS models (
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
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS events (
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
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS event_price_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    price REAL NOT NULL
                )
                """
            )
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT NOT NULL
                )
                """
            )
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_horizon ON events(horizon)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_event_prices_event_ts ON event_price_samples(event_id, ts)")
            self._conn.commit()

    def _event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["prediction"] = _json_load(out.pop("payload_json", "{}"), {})
        out["model_meta"] = _json_load(out.pop("model_meta_json", "{}"), {})
        out["metrics"] = _json_load(out.pop("metrics_json", "{}"), {})
        return out

    def _model_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["metrics_json"] = _json_load(out.get("metrics_json"), {})
        out["is_production"] = bool(out.get("is_production", 0))
        return out

    def upsert_model(self, payload: dict[str, Any]) -> None:
        metrics = payload.get("metrics_json", {})
        with self._lock:
            self._exec(
                """
                INSERT INTO models (
                    model_id, asset, horizon, name, created_at, git_commit, train_time,
                    dataset_hash, config_hash, metrics_json, is_production
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    asset=excluded.asset,
                    horizon=excluded.horizon,
                    name=excluded.name,
                    created_at=excluded.created_at,
                    git_commit=excluded.git_commit,
                    train_time=excluded.train_time,
                    dataset_hash=excluded.dataset_hash,
                    config_hash=excluded.config_hash,
                    metrics_json=excluded.metrics_json,
                    is_production=excluded.is_production
                """,
                (
                    str(payload["model_id"]),
                    str(payload.get("asset", "BTC")).upper(),
                    str(payload["horizon"]),
                    str(payload.get("name", payload["model_id"])),
                    str(payload.get("created_at", _to_iso_utc(_utc_now()))),
                    str(payload.get("git_commit", "unknown")),
                    str(payload.get("train_time", payload.get("created_at", _to_iso_utc(_utc_now())))),
                    str(payload.get("dataset_hash", "unknown")),
                    str(payload.get("config_hash", "unknown")),
                    _json_dump(metrics),
                    1 if bool(payload.get("is_production", False)) else 0,
                ),
            )
            self._conn.commit()

    def set_production_model(self, asset: str, horizon: str, model_id: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE models SET is_production=0 WHERE asset=? AND horizon=?",
                (asset.upper(), horizon),
            )
            self._exec(
                "UPDATE models SET is_production=1 WHERE asset=? AND horizon=? AND model_id=?",
                (asset.upper(), horizon, model_id),
            )
            self._conn.commit()

    def list_models(self, *, asset: str | None = None, horizon: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset:
            clauses.append("asset = ?")
            params.append(asset.upper())
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._exec(
                f"SELECT * FROM models {where} ORDER BY horizon ASC, created_at DESC",
                tuple(params),
            ).fetchall()
        return [self._model_from_row(row) for row in rows]

    def production_models(self, asset: str = "BTC") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                "SELECT * FROM models WHERE asset=? AND is_production=1 ORDER BY horizon ASC",
                (asset.upper(),),
            ).fetchall()
        return [self._model_from_row(row) for row in rows]

    def create_event(self, event: EventCreateInput) -> dict[str, Any]:
        if event.horizon not in HORIZON_TO_SECONDS:
            raise ValueError(f"unsupported horizon: {event.horizon}")
        event_id = str(uuid.uuid4())
        created_at = event.created_at.astimezone(timezone.utc)
        expires_at = created_at + timedelta(seconds=HORIZON_TO_SECONDS[event.horizon])
        now_iso = _to_iso_utc(_utc_now())

        with self._lock:
            self._exec(
                """
                INSERT INTO events (
                    event_id, asset, horizon, created_at, expires_at, price_t0, pred_low, pred_mid, pred_high,
                    payload_json, model_id, model_meta_json, price_source, status, actual_price, current_price,
                    metrics_json, note, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.asset.upper(),
                    event.horizon,
                    _to_iso_utc(created_at),
                    _to_iso_utc(expires_at),
                    float(event.price_t0),
                    float(event.pred_low),
                    float(event.pred_mid),
                    float(event.pred_high),
                    _json_dump(event.prediction),
                    event.model_id,
                    _json_dump(event.model_meta),
                    event.price_source,
                    "active",
                    None,
                    float(event.price_t0),
                    _json_dump({}),
                    event.note,
                    now_iso,
                ),
            )
            self._exec(
                "INSERT INTO event_price_samples (event_id, ts, price) VALUES (?, ?, ?)",
                (event_id, _to_iso_utc(created_at), float(event.price_t0)),
            )
            self._conn.commit()
        return self.get_event(event_id)

    def get_event(self, event_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._exec("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"event not found: {event_id}")
        return self._event_from_row(row)

    def list_active_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                "SELECT * FROM events WHERE status='active' ORDER BY created_at ASC"
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def list_events(
        self,
        *,
        status: str | None = None,
        horizon: str | None = None,
        model_id: str | None = None,
        result: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        q: str | None = None,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if created_from:
            clauses.append("created_at >= ?")
            params.append(created_from)
        if created_to:
            clauses.append("created_at <= ?")
            params.append(created_to)
        if q:
            clauses.append("(event_id LIKE ? OR note LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        if result == "hit":
            clauses.append("metrics_json LIKE '%\"hit\": true%'")
        elif result == "miss":
            clauses.append("metrics_json LIKE '%\"hit\": false%'")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        allowed_sort = {
            "created_at": "created_at",
            "expires_at": "expires_at",
            "price_t0": "price_t0",
            "current_price": "current_price",
            "status": "status",
            "horizon": "horizon",
        }
        sort_col = allowed_sort.get(sort_by, "created_at")
        sort_order = "ASC" if sort_dir.lower() == "asc" else "DESC"

        page_clean = max(1, int(page))
        page_size_clean = max(1, min(500, int(page_size)))
        offset = (page_clean - 1) * page_size_clean

        with self._lock:
            total = int(self._exec(f"SELECT COUNT(*) FROM events {where}", tuple(params)).fetchone()[0])
            rows = self._exec(
                f"""
                SELECT * FROM events
                {where}
                ORDER BY {sort_col} {sort_order}
                LIMIT ? OFFSET ?
                """,
                tuple([*params, page_size_clean, offset]),
            ).fetchall()

        items = [self._event_from_row(row) for row in rows]
        return {
            "items": items,
            "total": total,
            "page": page_clean,
            "page_size": page_size_clean,
        }

    def add_price_sample(self, event_id: str, ts: datetime, price: float) -> None:
        with self._lock:
            self._exec(
                "INSERT INTO event_price_samples (event_id, ts, price) VALUES (?, ?, ?)",
                (event_id, _to_iso_utc(ts), float(price)),
            )
            self._conn.commit()

    def list_price_samples(self, event_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                """
                SELECT id, event_id, ts, price
                FROM event_price_samples
                WHERE event_id=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (event_id, max(1, min(10000, int(limit)))),
            ).fetchall()
        out = [dict(row) for row in rows]
        out.reverse()
        return out

    def update_event_live(self, event_id: str, *, current_price: float, updated_at: datetime) -> None:
        with self._lock:
            self._exec(
                "UPDATE events SET current_price=?, updated_at=? WHERE event_id=?",
                (float(current_price), _to_iso_utc(updated_at), event_id),
            )
            self._conn.commit()

    def complete_event(
        self,
        event_id: str,
        *,
        actual_price: float,
        metrics: dict[str, Any],
        completed_at: datetime,
    ) -> None:
        with self._lock:
            self._exec(
                """
                UPDATE events
                SET status='completed',
                    actual_price=?,
                    current_price=?,
                    metrics_json=?,
                    updated_at=?
                WHERE event_id=?
                """,
                (
                    float(actual_price),
                    float(actual_price),
                    _json_dump(metrics),
                    _to_iso_utc(completed_at),
                    event_id,
                ),
            )
            self._conn.commit()

    def cancel_event(self, event_id: str, note: str | None = None) -> None:
        now = _to_iso_utc(_utc_now())
        metrics = {"cancelled_reason": note or "cancelled_by_user"}
        with self._lock:
            self._exec(
                "UPDATE events SET status='cancelled', metrics_json=?, updated_at=? WHERE event_id=?",
                (_json_dump(metrics), now, event_id),
            )
            self._conn.commit()

    def add_alert(self, level: str, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._exec(
                "INSERT INTO alerts (ts, level, code, message, context_json) VALUES (?, ?, ?, ?, ?)",
                (
                    _to_iso_utc(_utc_now()),
                    level,
                    code,
                    message,
                    _json_dump(context or {}),
                ),
            )
            self._conn.commit()

    def list_alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                "SELECT * FROM alerts ORDER BY ts DESC LIMIT ?",
                (max(1, min(1000, int(limit))),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["context_json"] = _json_load(item.get("context_json"), {})
            out.append(item)
        return out

    def metrics_summary(self, *, days: int = 30, horizon: str | None = None) -> dict[str, Any]:
        start_ts = _to_iso_utc(_utc_now() - timedelta(days=max(1, int(days))))
        clauses = ["status='completed'", "created_at >= ?"]
        params: list[Any] = [start_ts]
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        where = " AND ".join(clauses)

        with self._lock:
            rows = self._exec(
                f"SELECT horizon, created_at, metrics_json FROM events WHERE {where}",
                tuple(params),
            ).fetchall()
            active_count = int(
                self._exec("SELECT COUNT(*) FROM events WHERE status='active'").fetchone()[0]
            )

        items = []
        for row in rows:
            metrics = _json_load(row["metrics_json"], {})
            items.append(
                {
                    "horizon": row["horizon"],
                    "created_at": row["created_at"],
                    "hit": bool(metrics.get("hit", False)),
                    "abs_error": float(metrics.get("abs_error", 0.0)),
                    "rel_error": float(metrics.get("rel_error", 0.0)),
                    "width_pct": float(metrics.get("width_pct", 0.0)),
                }
            )

        completed = len(items)
        if completed == 0:
            return {
                "period_days": int(days),
                "target_coverage": 0.80,
                "actual_coverage": 0.0,
                "coverage_delta": -0.80,
                "completed_events": 0,
                "active_events": active_count,
                "avg_width_pct": 0.0,
                "avg_abs_error": 0.0,
                "avg_rel_error": 0.0,
                "best_horizon": None,
                "series": [],
                "by_horizon": {},
            }

        hits = sum(1 for item in items if item["hit"])
        actual_coverage = hits / completed
        avg_width = sum(item["width_pct"] for item in items) / completed
        avg_abs_error = sum(item["abs_error"] for item in items) / completed
        avg_rel_error = sum(item["rel_error"] for item in items) / completed

        by_horizon: dict[str, dict[str, Any]] = {}
        for item in items:
            horizon_key = item["horizon"]
            group = by_horizon.setdefault(
                horizon_key,
                {"count": 0, "hits": 0, "width_sum": 0.0, "abs_error_sum": 0.0},
            )
            group["count"] += 1
            group["hits"] += 1 if item["hit"] else 0
            group["width_sum"] += item["width_pct"]
            group["abs_error_sum"] += item["abs_error"]
        for horizon_key, group in by_horizon.items():
            count = max(1, group["count"])
            group["coverage"] = group["hits"] / count
            group["avg_width_pct"] = group["width_sum"] / count
            group["avg_abs_error"] = group["abs_error_sum"] / count
            del group["width_sum"]
            del group["abs_error_sum"]

        best_horizon = max(
            by_horizon.items(),
            key=lambda pair: (pair[1]["coverage"], pair[1]["count"]),
        )[0]

        series_map: dict[str, dict[str, Any]] = {}
        for item in items:
            day = item["created_at"][:10]
            slot = series_map.setdefault(day, {"date": day, "count": 0, "hits": 0, "width_sum": 0.0})
            slot["count"] += 1
            slot["hits"] += 1 if item["hit"] else 0
            slot["width_sum"] += item["width_pct"]
        series = []
        for day in sorted(series_map.keys()):
            slot = series_map[day]
            count = max(1, slot["count"])
            series.append(
                {
                    "date": day,
                    "count": slot["count"],
                    "coverage": slot["hits"] / count,
                    "width_pct": slot["width_sum"] / count,
                }
            )

        return {
            "period_days": int(days),
            "target_coverage": 0.80,
            "actual_coverage": actual_coverage,
            "coverage_delta": actual_coverage - 0.80,
            "completed_events": completed,
            "active_events": active_count,
            "avg_width_pct": avg_width,
            "avg_abs_error": avg_abs_error,
            "avg_rel_error": avg_rel_error,
            "best_horizon": best_horizon,
            "series": series,
            "by_horizon": by_horizon,
        }

    @staticmethod
    def parse_utc(ts: str) -> datetime:
        return _parse_utc(ts)
