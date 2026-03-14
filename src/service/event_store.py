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

from src.analysis.tournament_eval import build_tournament_summary

HORIZON_TO_SECONDS: dict[str, int] = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}

LEGACY_TO_JOB_TYPE: dict[str, str] = {
    "train": "train_model",
    "sweep": "sweep_train",
    "train_model": "train_model",
    "sweep_train": "sweep_train",
    "wf_eval": "wf_eval",
}

JOB_TYPE_TO_LEGACY: dict[str, str] = {
    "train_model": "train",
    "sweep_train": "sweep",
    "wf_eval": "wf_eval",
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


_UNSET = object()


def _bucket_floor_utc(ts: datetime, horizon: str) -> datetime:
    seconds = int(HORIZON_TO_SECONDS[horizon])
    ts_utc = ts.astimezone(timezone.utc)
    floored = int(ts_utc.timestamp()) // seconds * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _build_event_eval_key(asset: str, horizon: str, created_at: datetime) -> str:
    bucket_ts = _bucket_floor_utc(created_at, horizon)
    return f"{asset.upper()}|{horizon}|{_to_iso_utc(bucket_ts)}"


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
    cycle_id: str | None = None
    cycle_seq: int | None = None
    cycle_total: int | None = None
    params_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class CycleCreateInput:
    asset: str
    horizon: str
    total_runs: int
    model_id: str | None
    price_source: str
    mode: str = "single"
    model_ids: tuple[str, ...] = ()
    models_count: int = 1
    align_to_boundary: bool = False
    note: str | None = None


class EventStore:
    """Thread-safe SQLite storage used by the dashboard backend."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._init_schema()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if self._closed:
            raise RuntimeError("event store is closed")
        return self._conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __del__(self) -> None:  # pragma: no cover - best-effort finalizer.
        try:
            self.close()
        except Exception:
            pass

    def _has_column(self, table: str, column: str) -> bool:
        rows = self._exec(f"PRAGMA table_info({table})").fetchall()
        return any(str(row["name"]) == column for row in rows)

    def _has_table(self, table: str) -> bool:
        row = self._exec(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        if self._has_column(table, column):
            return
        self._exec(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

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
                    is_production INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    deleted_at TEXT,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    notes TEXT,
                    tags_json TEXT NOT NULL DEFAULT '{}',
                    parent_model_id TEXT,
                    experiment_id TEXT
                )
                """
            )
            self._ensure_column("models", "status", "TEXT NOT NULL DEFAULT 'active'")
            self._ensure_column("models", "deleted_at", "TEXT")
            self._ensure_column("models", "params_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("models", "notes", "TEXT")
            self._ensure_column("models", "tags_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("models", "parent_model_id", "TEXT")
            self._ensure_column("models", "experiment_id", "TEXT")
            self._exec("UPDATE models SET params_json='{}' WHERE params_json IS NULL")
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
                    params_json TEXT NOT NULL DEFAULT '{}',
                    eval_key TEXT NOT NULL DEFAULT '',
                    note TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column("events", "cycle_id", "TEXT")
            self._ensure_column("events", "cycle_seq", "INTEGER")
            self._ensure_column("events", "cycle_total", "INTEGER")
            self._ensure_column("events", "params_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("events", "eval_key", "TEXT")
            self._exec("UPDATE events SET params_json='{}' WHERE params_json IS NULL")
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    total_runs INTEGER NOT NULL,
                    launched_runs INTEGER NOT NULL DEFAULT 0,
                    completed_runs INTEGER NOT NULL DEFAULT 0,
                    cancelled_runs INTEGER NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL DEFAULT 'single',
                    model_id TEXT,
                    model_ids_json TEXT,
                    models_count INTEGER NOT NULL DEFAULT 1,
                    align_to_boundary INTEGER NOT NULL DEFAULT 0,
                    price_source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    note TEXT,
                    last_event_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column("cycles", "mode", "TEXT NOT NULL DEFAULT 'single'")
            self._ensure_column("cycles", "model_ids_json", "TEXT")
            self._ensure_column("cycles", "models_count", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("cycles", "align_to_boundary", "INTEGER NOT NULL DEFAULT 0")
            self._exec("UPDATE cycles SET mode='single' WHERE mode IS NULL OR mode=''")
            self._exec("UPDATE cycles SET models_count=1 WHERE models_count IS NULL OR models_count <= 0")
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
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
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
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS training_job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    stage TEXT,
                    message TEXT NOT NULL
                )
                """
            )
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    progress REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_text TEXT,
                    logs_text TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._exec(
                """
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    stage TEXT,
                    message TEXT NOT NULL
                )
                """
            )
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_horizon ON events(horizon)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_cycle_id ON events(cycle_id)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_events_eval_key ON events(eval_key)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_cycles_status ON cycles(status)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_event_prices_event_ts ON event_price_samples(event_id, ts)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_models_asset_horizon_status ON models(asset, horizon, status)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created ON training_jobs(status, created_at)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_training_job_logs_job_ts ON training_job_logs(job_id, ts)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)")
            self._exec("CREATE INDEX IF NOT EXISTS idx_job_logs_job_ts ON job_logs(job_id, ts)")
            self._migrate_legacy_jobs_locked()
            self._backfill_event_eval_keys_locked(only_missing=True)
            self._conn.commit()

    def _event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["prediction"] = _json_load(out.pop("payload_json", "{}"), {})
        out["model_meta"] = _json_load(out.pop("model_meta_json", "{}"), {})
        out["metrics"] = _json_load(out.pop("metrics_json", "{}"), {})
        out["params_json"] = _json_load(out.pop("params_json", "{}"), {})
        return out

    def _model_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["metrics_json"] = _json_load(out.get("metrics_json"), {})
        out["params_json"] = _json_load(out.get("params_json"), {})
        out["is_production"] = bool(out.get("is_production", 0))
        out["tags"] = _json_load(out.pop("tags_json", "{}"), {})
        return out

    def _cycle_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["mode"] = str(out.get("mode", "single") or "single")
        out["model_ids"] = _json_load(out.pop("model_ids_json", "[]"), [])
        out["models_count"] = max(1, int(out.get("models_count", 1) or 1))
        out["align_to_boundary"] = bool(out.get("align_to_boundary", 0))
        return out

    def _job_type_from_legacy(self, value: str) -> str:
        raw = str(value or "").strip().lower()
        return LEGACY_TO_JOB_TYPE.get(raw, "train_model")

    def _legacy_job_kind_from_type(self, value: str) -> str:
        raw = str(value or "").strip().lower()
        return JOB_TYPE_TO_LEGACY.get(raw, raw or "train")

    def _job_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["type"] = str(out.get("type", "train_model"))
        out["params"] = _json_load(out.pop("params_json", "{}"), {})
        out["result"] = _json_load(out.pop("result_json", "{}"), {})
        out["progress"] = max(0.0, min(1.0, float(out.get("progress", 0.0) or 0.0)))
        out["logs_text"] = str(out.get("logs_text", "") or "")
        params = out.get("params", {})
        if isinstance(params, dict):
            out["asset"] = str(params.get("asset", "BTC")).upper()
            out["horizons"] = [str(item) for item in params.get("horizons", []) if item]
            out["cancel_requested"] = bool(params.get("cancel_requested", False))
        else:
            out["asset"] = "BTC"
            out["horizons"] = []
            out["cancel_requested"] = False
        stage = str(out.get("stage", "queued") or "queued").strip().lower()
        allowed_stages = {"queued", "data", "features", "train", "calibrate", "register", "done"}
        if stage not in allowed_stages:
            stage = "done" if str(out.get("status", "")).lower() in {"succeeded", "failed", "canceled"} else "queued"
        out["stage"] = stage
        out["summary"] = self._job_summary_from_payload(out)
        return out

    def _training_job_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["horizons"] = _json_load(out.pop("horizons_json", "[]"), [])
        out["params"] = _json_load(out.pop("params_json", "{}"), {})
        out["result"] = _json_load(out.pop("result_json", "{}"), {})
        out["cancel_requested"] = bool(out.get("cancel_requested", 0))
        return out

    def _legacy_training_job_from_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(payload["job_id"]),
            "job_kind": self._legacy_job_kind_from_type(str(payload.get("type", "train_model"))),
            "asset": str(payload.get("asset", "BTC")).upper(),
            "horizons": [str(item) for item in payload.get("horizons", [])],
            "status": str(payload.get("status", "pending")),
            "progress": int(round(float(payload.get("progress", 0.0)) * 100)),
            "stage": str(payload.get("stage", "queued")),
            "summary": str(payload.get("summary", self._job_summary_from_payload(payload))),
            "params": dict(payload.get("params", {})),
            "result": dict(payload.get("result", {})),
            "error": payload.get("error_text"),
            "cancel_requested": bool(payload.get("cancel_requested", False)),
            "created_at": str(payload.get("created_at")),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
        }

    def _job_summary_from_payload(self, payload: dict[str, Any]) -> str:
        params = dict(payload.get("params", {}))
        summary_hint = str(params.get("job_summary_hint", "")).strip()
        if summary_hint:
            return summary_hint
        asset = str(payload.get("asset", params.get("asset", "BTC"))).upper()
        horizons = [str(item) for item in payload.get("horizons", params.get("horizons", [])) if item]
        base_mode = str(params.get("base_timeframe_mode", "legacy"))
        quantiles = [str(item) for item in payload.get("result", {}).get("quantiles_final", params.get("selected_quantiles", [])) if item]
        quantiles_text = "/".join(quantiles) if quantiles else "-"
        return f"{payload.get('type', 'train_model')} {asset} {', '.join(horizons) or '-'} {base_mode} {quantiles_text}"

    def _migrate_legacy_jobs_locked(self) -> None:
        if not self._has_table("training_jobs") or not self._has_table("jobs"):
            return
        legacy_count = int(self._exec("SELECT COUNT(*) FROM training_jobs").fetchone()[0])
        jobs_count = int(self._exec("SELECT COUNT(*) FROM jobs").fetchone()[0])
        if legacy_count <= 0 or jobs_count > 0:
            return
        legacy_rows = self._exec("SELECT * FROM training_jobs ORDER BY created_at ASC").fetchall()
        for row in legacy_rows:
            legacy = self._training_job_from_row(row)
            params = dict(legacy.get("params", {}))
            summary_hint = str(legacy.get("summary", "")).strip()
            if summary_hint and not str(params.get("job_summary_hint", "")).strip():
                params["job_summary_hint"] = summary_hint
            params["cancel_requested"] = bool(legacy.get("cancel_requested", False))
            self._exec(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, type, status, created_at, started_at, finished_at,
                    progress, stage, params_json, result_json, error_text, logs_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(legacy["job_id"]),
                    self._job_type_from_legacy(str(legacy.get("job_kind", "train"))),
                    str(legacy.get("status", "pending")),
                    str(legacy.get("created_at")),
                    legacy.get("started_at"),
                    legacy.get("finished_at"),
                    max(0.0, min(1.0, float(legacy.get("progress", 0)) / 100.0)),
                    str(legacy.get("stage", "queued")),
                    _json_dump(params),
                    _json_dump(legacy.get("result", {})),
                    legacy.get("error"),
                    "",
                ),
            )
        if self._has_table("training_job_logs") and self._has_table("job_logs"):
            log_count = int(self._exec("SELECT COUNT(*) FROM job_logs").fetchone()[0])
            if log_count <= 0:
                legacy_logs = self._exec(
                    "SELECT job_id, ts, level, stage, message FROM training_job_logs ORDER BY ts ASC, id ASC"
                ).fetchall()
                for row in legacy_logs:
                    self._exec(
                        """
                        INSERT INTO job_logs (job_id, ts, level, stage, message)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(row["job_id"]),
                            str(row["ts"]),
                            str(row["level"]),
                            row["stage"],
                            str(row["message"]),
                        ),
                    )
                    existing = self._exec("SELECT logs_text FROM jobs WHERE job_id=?", (str(row["job_id"]),)).fetchone()
                    if existing is None:
                        continue
                    prefix = str(existing["logs_text"] or "")
                    line = f"[{row['ts']}] {str(row['level']).upper()}{'/' + str(row['stage']) if row['stage'] else ''} {row['message']}"
                    next_text = f"{prefix}\n{line}".strip()
                    self._exec("UPDATE jobs SET logs_text=? WHERE job_id=?", (next_text, str(row["job_id"])))

    def _append_job_log_locked(
        self,
        job_id: str,
        *,
        ts: str,
        level: str,
        stage: str | None,
        message: str,
    ) -> None:
        row = self._exec("SELECT logs_text FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        self._exec(
            """
            INSERT INTO job_logs (job_id, ts, level, stage, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, ts, level, stage, message),
        )
        prefix = str(row["logs_text"] or "")
        suffix = f"[{ts}] {str(level).upper()}{'/' + str(stage) if stage else ''} {message}"
        next_text = f"{prefix}\n{suffix}".strip()
        self._exec("UPDATE jobs SET logs_text=? WHERE job_id=?", (next_text, job_id))

    def _event_params_snapshot_locked(self, event: EventCreateInput) -> dict[str, Any]:
        if isinstance(event.params_snapshot, dict) and event.params_snapshot:
            return dict(event.params_snapshot)
        row = self._exec(
            "SELECT params_json FROM models WHERE model_id=?",
            (event.model_id,),
        ).fetchone()
        if row is not None:
            payload = _json_load(row["params_json"], {})
            if isinstance(payload, dict) and payload:
                return payload
        return {
            "asset": event.asset.upper(),
            "horizon": event.horizon,
            "model_id": event.model_id,
            "price_source": event.price_source,
        }

    def _backfill_event_eval_keys_locked(self, *, only_missing: bool) -> int:
        where = "WHERE eval_key IS NULL OR eval_key=''" if only_missing else ""
        rows = self._exec(
            f"SELECT event_id, asset, horizon, created_at, eval_key FROM events {where}"
        ).fetchall()
        updated = 0
        for row in rows:
            horizon = str(row["horizon"])
            if horizon not in HORIZON_TO_SECONDS:
                continue
            created_at = _parse_utc(str(row["created_at"]))
            next_key = _build_event_eval_key(str(row["asset"]), horizon, created_at)
            if str(row["eval_key"] or "") == next_key:
                continue
            self._exec("UPDATE events SET eval_key=? WHERE event_id=?", (next_key, str(row["event_id"])))
            updated += 1
        return updated

    def backfill_event_eval_keys(self, *, only_missing: bool = False) -> int:
        with self._lock:
            updated = self._backfill_event_eval_keys_locked(only_missing=only_missing)
            self._conn.commit()
        return updated

    def upsert_model(self, payload: dict[str, Any]) -> None:
        metrics = payload.get("metrics_json", {})
        params_json = payload.get("params_json", {})
        tags = payload.get("tags", {})
        with self._lock:
            self._exec(
                """
                INSERT INTO models (
                    model_id, asset, horizon, name, created_at, git_commit, train_time,
                    dataset_hash, config_hash, metrics_json, is_production, status, deleted_at, params_json, notes, tags_json,
                    parent_model_id, experiment_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    is_production=excluded.is_production,
                    status=models.status,
                    deleted_at=CASE
                        WHEN models.status='deleted' THEN COALESCE(models.deleted_at, excluded.deleted_at)
                        ELSE excluded.deleted_at
                    END,
                    params_json=excluded.params_json,
                    notes=CASE WHEN models.notes IS NOT NULL THEN models.notes ELSE excluded.notes END,
                    tags_json=CASE WHEN models.tags_json IS NOT NULL AND models.tags_json != '{}' THEN models.tags_json ELSE excluded.tags_json END,
                    parent_model_id=CASE WHEN models.parent_model_id IS NOT NULL THEN models.parent_model_id ELSE excluded.parent_model_id END,
                    experiment_id=CASE WHEN models.experiment_id IS NOT NULL THEN models.experiment_id ELSE excluded.experiment_id END
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
                    str(payload.get("status", "active")),
                    payload.get("deleted_at"),
                    _json_dump(params_json),
                    payload.get("notes"),
                    _json_dump(tags),
                    payload.get("parent_model_id"),
                    payload.get("experiment_id"),
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

    def list_models(
        self,
        *,
        asset: str | None = None,
        horizon: str | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset:
            clauses.append("asset = ?")
            params.append(asset.upper())
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        if statuses:
            cleaned = [str(item) for item in statuses if item]
            if cleaned:
                placeholders = ", ".join("?" for _ in cleaned)
                clauses.append(f"status IN ({placeholders})")
                params.extend(cleaned)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(1, min(1000, int(limit))))
        with self._lock:
            rows = self._exec(
                f"SELECT * FROM models {where} ORDER BY created_at DESC{limit_sql}",
                tuple(params),
            ).fetchall()
            counts = {
                str(item["model_id"]): int(item["refs"])
                for item in self._exec(
                    "SELECT model_id, COUNT(*) AS refs FROM events GROUP BY model_id"
                ).fetchall()
            }
        out = [self._model_from_row(row) for row in rows]
        for item in out:
            item["event_refs"] = counts.get(str(item["model_id"]), 0)
        return out

    def production_models(self, asset: str = "BTC") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                "SELECT * FROM models WHERE asset=? AND is_production=1 AND status='active' ORDER BY horizon ASC",
                (asset.upper(),),
            ).fetchall()
            counts = {
                str(item["model_id"]): int(item["refs"])
                for item in self._exec(
                    "SELECT model_id, COUNT(*) AS refs FROM events GROUP BY model_id"
                ).fetchall()
            }
        out = [self._model_from_row(row) for row in rows]
        for item in out:
            item["event_refs"] = counts.get(str(item["model_id"]), 0)
        return out

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._exec("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
            refs = int(
                self._exec(
                    "SELECT COUNT(*) FROM events WHERE model_id=?",
                    (model_id,),
                ).fetchone()[0]
            )
        if row is None:
            raise KeyError(f"model not found: {model_id}")
        out = self._model_from_row(row)
        out["event_refs"] = refs
        return out

    def update_model_management(
        self,
        model_id: str,
        *,
        status: str | None = None,
        notes: str | None | object = _UNSET,
        tags: dict[str, str] | object = _UNSET,
        parent_model_id: str | None | object = _UNSET,
        experiment_id: str | None | object = _UNSET,
    ) -> dict[str, Any]:
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "deleted":
                updates.append("deleted_at = ?")
                params.append(_to_iso_utc(_utc_now()))
            elif status in {"active", "archived"}:
                updates.append("deleted_at = ?")
                params.append(None)
        if notes is not _UNSET:
            updates.append("notes = ?")
            params.append(notes)
        if tags is not _UNSET:
            updates.append("tags_json = ?")
            params.append(_json_dump(tags))
        if parent_model_id is not _UNSET:
            updates.append("parent_model_id = ?")
            params.append(parent_model_id)
        if experiment_id is not _UNSET:
            updates.append("experiment_id = ?")
            params.append(experiment_id)
        if not updates:
            return self.get_model(model_id)
        with self._lock:
            updated = self._exec(
                f"UPDATE models SET {', '.join(updates)} WHERE model_id=?",
                tuple([*params, model_id]),
            )
            self._conn.commit()
        if updated.rowcount <= 0:
            raise KeyError(f"model not found: {model_id}")
        return self.get_model(model_id)

    def count_events_for_model(self, model_id: str) -> int:
        with self._lock:
            return int(
                self._exec(
                    "SELECT COUNT(*) FROM events WHERE model_id=?",
                    (model_id,),
                ).fetchone()[0]
            )

    def purge_model_row(self, model_id: str) -> None:
        with self._lock:
            deleted = self._exec("DELETE FROM models WHERE model_id=?", (model_id,))
            self._conn.commit()
        if deleted.rowcount <= 0:
            raise KeyError(f"model not found: {model_id}")

    def create_job(
        self,
        *,
        job_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now_iso = _to_iso_utc(_utc_now())
        payload = dict(params)
        with self._lock:
            self._exec(
                """
                INSERT INTO jobs (
                    job_id, type, status, created_at, started_at, finished_at,
                    progress, stage, params_json, result_json, error_text, logs_text
                )
                VALUES (?, ?, 'pending', ?, NULL, NULL, 0.0, 'queued', ?, '{}', NULL, '')
                """,
                (
                    job_id,
                    str(job_type).strip().lower() or "train_model",
                    now_iso,
                    _json_dump(payload),
                ),
            )
            self._append_job_log_locked(
                job_id,
                ts=now_iso,
                level="info",
                stage="queued",
                message="job created",
            )
            self._conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._exec("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return self._job_from_row(row)

    def list_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        job_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            cleaned_statuses = [str(item).strip().lower() for item in statuses if str(item).strip()]
            if cleaned_statuses:
                placeholders = ", ".join("?" for _ in cleaned_statuses)
                clauses.append(f"status IN ({placeholders})")
                params.extend(cleaned_statuses)
        if job_types:
            cleaned_types = [str(item).strip().lower() for item in job_types if str(item).strip()]
            if cleaned_types:
                placeholders = ", ".join("?" for _ in cleaned_types)
                clauses.append(f"type IN ({placeholders})")
                params.extend(cleaned_types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._exec(
                f"""
                SELECT * FROM jobs
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple([*params, max(1, min(1000, int(limit)))]),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def add_job_log(self, job_id: str, *, message: str, level: str = "info", stage: str | None = None) -> None:
        now_iso = _to_iso_utc(_utc_now())
        with self._lock:
            self._append_job_log_locked(
                job_id,
                ts=now_iso,
                level=level,
                stage=stage,
                message=message,
            )
            self._conn.commit()

    def list_job_logs(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                """
                SELECT * FROM job_logs
                WHERE job_id=?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (job_id, max(1, min(5000, int(limit)))),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_next_job(self, *, job_types: list[str] | None = None) -> dict[str, Any] | None:
        now_iso = _to_iso_utc(_utc_now())
        clauses = ["status='pending'"]
        params: list[Any] = []
        if job_types:
            cleaned_types = [str(item).strip().lower() for item in job_types if str(item).strip()]
            if cleaned_types:
                placeholders = ", ".join("?" for _ in cleaned_types)
                clauses.append(f"type IN ({placeholders})")
                params.extend(cleaned_types)
        where = " AND ".join(clauses)
        with self._lock:
            row = self._exec(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY created_at ASC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            self._exec(
                """
                UPDATE jobs
                SET status='running',
                    progress=?,
                    stage='data',
                    started_at=COALESCE(started_at, ?)
                WHERE job_id=?
                """,
                (max(0.0, min(1.0, 0.01)), now_iso, job_id),
            )
            self._append_job_log_locked(
                job_id,
                ts=now_iso,
                level="info",
                stage="data",
                message="job claimed by worker",
            )
            self._conn.commit()
            claimed = self._exec("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job_from_row(claimed) if claimed is not None else None

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        stage: str | None = None,
        params: dict[str, Any] | object = _UNSET,
        result: dict[str, Any] | None = None,
        error_text: str | None | object = _UNSET,
        finished: bool = False,
    ) -> dict[str, Any]:
        updates: list[str] = []
        sql_params: list[Any] = []
        next_stage = stage
        terminal_statuses = {"succeeded", "failed", "canceled"}
        if next_stage is None and (finished or (status is not None and str(status).strip().lower() in terminal_statuses)):
            next_stage = "done"
        if status is not None:
            updates.append("status = ?")
            sql_params.append(str(status).strip().lower())
        if progress is not None:
            updates.append("progress = ?")
            sql_params.append(max(0.0, min(1.0, float(progress))))
        if next_stage is not None:
            updates.append("stage = ?")
            sql_params.append(next_stage)
        if params is not _UNSET:
            updates.append("params_json = ?")
            sql_params.append(_json_dump(dict(params)))
        if result is not None:
            updates.append("result_json = ?")
            sql_params.append(_json_dump(result))
        if error_text is not _UNSET:
            updates.append("error_text = ?")
            sql_params.append(error_text)
        if finished:
            updates.append("finished_at = ?")
            sql_params.append(_to_iso_utc(_utc_now()))
        if not updates:
            return self.get_job(job_id)
        with self._lock:
            updated = self._exec(
                f"UPDATE jobs SET {', '.join(updates)} WHERE job_id=?",
                tuple([*sql_params, job_id]),
            )
            self._conn.commit()
        if updated.rowcount <= 0:
            raise KeyError(f"job not found: {job_id}")
        return self.get_job(job_id)

    def is_job_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._exec("SELECT params_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        params = _json_load(row["params_json"], {})
        return bool(dict(params).get("cancel_requested", False)) if isinstance(params, dict) else False

    def request_cancel_job(self, job_id: str) -> dict[str, Any]:
        now_iso = _to_iso_utc(_utc_now())
        with self._lock:
            row = self._exec("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"job not found: {job_id}")
            current = self._job_from_row(row)
            status = str(current.get("status", "pending")).lower()
            if status in {"succeeded", "failed", "canceled"}:
                return current
            params = dict(current.get("params", {}))
            params["cancel_requested"] = True
            if status == "pending":
                self._exec(
                    """
                    UPDATE jobs
                    SET status='canceled',
                        progress=1.0,
                        stage='done',
                        params_json=?,
                        error_text=?,
                        finished_at=?
                    WHERE job_id=?
                    """,
                    (_json_dump(params), "job canceled before start", now_iso, job_id),
                )
                log_level = "warning"
                log_stage = "done"
            else:
                self._exec(
                    "UPDATE jobs SET params_json=? WHERE job_id=?",
                    (_json_dump(params), job_id),
                )
                log_level = "warning"
                log_stage = str(current.get("stage", "data"))
            self._append_job_log_locked(
                job_id,
                ts=now_iso,
                level=log_level,
                stage=log_stage,
                message="cancel requested",
            )
            self._conn.commit()
        return self.get_job(job_id)

    def create_training_job(
        self,
        *,
        job_kind: str,
        asset: str,
        horizons: list[str],
        summary: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(params)
        payload["asset"] = str(asset).upper()
        payload["horizons"] = [str(item) for item in horizons if item]
        payload["job_summary_hint"] = str(summary).strip()
        job = self.create_job(
            job_type=self._job_type_from_legacy(job_kind),
            params=payload,
        )
        return self._legacy_training_job_from_job(job)

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if str(job.get("type")) not in {"train_model", "sweep_train", "wf_eval"}:
            raise KeyError(f"training job not found: {job_id}")
        return self._legacy_training_job_from_job(job)

    def list_training_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        jobs = self.list_jobs(
            statuses=statuses,
            job_types=["train_model", "sweep_train", "wf_eval"],
            limit=limit,
        )
        return [self._legacy_training_job_from_job(job) for job in jobs]

    def add_training_job_log(self, job_id: str, *, message: str, level: str = "info", stage: str | None = None) -> None:
        self.add_job_log(job_id, message=message, level=level, stage=stage)

    def list_training_job_logs(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        return self.list_job_logs(job_id, limit=limit)

    def claim_next_training_job(self) -> dict[str, Any] | None:
        claimed = self.claim_next_job(job_types=["train_model", "sweep_train", "wf_eval"])
        if claimed is None:
            return None
        return self._legacy_training_job_from_job(claimed)

    def update_training_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        stage: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None | object = _UNSET,
        finished: bool = False,
    ) -> dict[str, Any]:
        job = self.update_job(
            job_id,
            status=status,
            progress=None if progress is None else (max(0, min(100, int(progress))) / 100.0),
            stage=stage,
            result=result,
            error_text=error,
            finished=finished,
        )
        return self._legacy_training_job_from_job(job)

    def is_training_job_cancel_requested(self, job_id: str) -> bool:
        return self.is_job_cancel_requested(job_id)

    def request_cancel_training_job(self, job_id: str) -> dict[str, Any]:
        job = self.request_cancel_job(job_id)
        if str(job.get("type")) not in {"train_model", "sweep_train", "wf_eval"}:
            raise KeyError(f"training job not found: {job_id}")
        return self._legacy_training_job_from_job(job)

    def create_cycle(self, cycle: CycleCreateInput) -> dict[str, Any]:
        if cycle.horizon not in HORIZON_TO_SECONDS:
            raise ValueError(f"unsupported horizon: {cycle.horizon}")
        total_runs = max(1, int(cycle.total_runs))
        cycle_id = str(uuid.uuid4())
        now_iso = _to_iso_utc(_utc_now())
        mode = str(cycle.mode or "single").strip().lower()
        model_ids = [str(item) for item in cycle.model_ids if str(item)]
        models_count = max(1, int(cycle.models_count or (len(model_ids) or 1)))
        with self._lock:
            self._exec(
                """
                INSERT INTO cycles (
                    cycle_id, asset, horizon, total_runs, launched_runs, completed_runs, cancelled_runs,
                    mode, model_id, model_ids_json, models_count, align_to_boundary,
                    price_source, status, note, last_event_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, ?, ?, ?, ?, ?, ?, 'running', ?, NULL, ?, ?)
                """,
                (
                    cycle_id,
                    cycle.asset.upper(),
                    cycle.horizon,
                    total_runs,
                    mode,
                    cycle.model_id,
                    _json_dump(model_ids),
                    models_count,
                    1 if bool(cycle.align_to_boundary) else 0,
                    cycle.price_source,
                    cycle.note,
                    now_iso,
                    now_iso,
                ),
            )
            self._conn.commit()
        return self.get_cycle(cycle_id)

    def get_cycle(self, cycle_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._exec("SELECT * FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
        if row is None:
            raise KeyError(f"cycle not found: {cycle_id}")
        return self._cycle_from_row(row)

    def list_cycles(
        self,
        *,
        status: str | None = None,
        horizon: str | None = None,
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
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        page_clean = max(1, int(page))
        page_size_clean = max(1, min(500, int(page_size)))
        offset = (page_clean - 1) * page_size_clean
        with self._lock:
            total = int(self._exec(f"SELECT COUNT(*) FROM cycles {where}", tuple(params)).fetchone()[0])
            rows = self._exec(
                f"""
                SELECT * FROM cycles
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple([*params, page_size_clean, offset]),
            ).fetchall()
        return {
            "items": [self._cycle_from_row(row) for row in rows],
            "total": total,
            "page": page_clean,
            "page_size": page_size_clean,
        }

    def list_running_cycles(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._exec(
                """
                SELECT * FROM cycles
                WHERE status='running' AND launched_runs < total_runs
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max(1, min(1000, int(limit))),),
            ).fetchall()
        return [self._cycle_from_row(row) for row in rows]

    def cycle_summary(
        self,
        cycle_id: str,
        *,
        orchestrator_rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            cycle_row = self._exec("SELECT * FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
            if cycle_row is None:
                raise KeyError(f"cycle not found: {cycle_id}")
            cycle = self._cycle_from_row(cycle_row)
            event_rows = self._exec(
                """
                SELECT * FROM events
                WHERE cycle_id=?
                ORDER BY cycle_seq ASC, created_at ASC, event_id ASC
                """,
                (cycle_id,),
            ).fetchall()
            events = [self._event_from_row(row) for row in event_rows]

            model_ids: list[str] = []
            for raw in list(cycle.get("model_ids", [])) + [cycle.get("model_id")]:
                model_id = str(raw or "").strip()
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)
            for event in events:
                model_id = str(event.get("model_id", "")).strip()
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)

            model_lookup: dict[str, dict[str, Any]] = {}
            if model_ids:
                placeholders = ", ".join("?" for _ in model_ids)
                rows = self._exec(
                    f"SELECT * FROM models WHERE model_id IN ({placeholders})",
                    tuple(model_ids),
                ).fetchall()
                model_lookup = {
                    str(row["model_id"]): self._model_from_row(row)
                    for row in rows
                }

        return build_tournament_summary(
            cycle=cycle,
            events=events,
            model_lookup=model_lookup,
            rules_override=orchestrator_rules,
        )

    def apply_tournament_early_stop(self, cycle_id: str) -> bool:
        now_iso = _to_iso_utc(_utc_now())
        changed = False
        with self._lock:
            cycle_row = self._exec("SELECT * FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
            if cycle_row is None:
                raise KeyError(f"cycle not found: {cycle_id}")
            cycle = self._cycle_from_row(cycle_row)
            if str(cycle.get("mode", "single")) != "tournament" or str(cycle.get("status", "running")) != "running":
                return False

            limited = self._exec(
                """
                UPDATE cycles
                SET total_runs = launched_runs,
                    updated_at = ?
                WHERE cycle_id = ?
                  AND status = 'running'
                  AND total_runs > launched_runs
                """,
                (now_iso, cycle_id),
            )
            changed = limited.rowcount > 0

            refreshed = self._exec(
                "SELECT launched_runs, completed_runs, cancelled_runs, status FROM cycles WHERE cycle_id=?",
                (cycle_id,),
            ).fetchone()
            if refreshed is not None and str(refreshed["status"]) == "running":
                active_count = int(
                    self._exec(
                        "SELECT COUNT(*) FROM events WHERE cycle_id=? AND status='active'",
                        (cycle_id,),
                    ).fetchone()[0]
                )
                launched_runs = int(refreshed["launched_runs"])
                accounted_runs = int(refreshed["completed_runs"]) + int(refreshed["cancelled_runs"])
                if active_count <= 0 and accounted_runs >= launched_runs:
                    completed = self._exec(
                        """
                        UPDATE cycles
                        SET status='completed',
                            updated_at=?
                        WHERE cycle_id=?
                          AND status='running'
                        """,
                        (now_iso, cycle_id),
                    )
                    changed = changed or (completed.rowcount > 0)

            if changed:
                self._conn.commit()
        return changed

    def has_active_event_for_cycle(self, cycle_id: str) -> bool:
        with self._lock:
            count = int(
                self._exec(
                    "SELECT COUNT(*) FROM events WHERE cycle_id=? AND status='active'",
                    (cycle_id,),
                ).fetchone()[0]
            )
        return count > 0

    def _validate_cycle_for_new_run_locked(self, cycle_id: str) -> dict[str, Any]:
        row = self._exec(
            "SELECT * FROM cycles WHERE cycle_id=?",
            (cycle_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"cycle not found: {cycle_id}")
        cycle = self._cycle_from_row(row)
        if str(cycle["status"]) != "running":
            raise ValueError(f"cycle is not running: {cycle_id}")
        if int(cycle["launched_runs"]) >= int(cycle["total_runs"]):
            raise ValueError(f"cycle is already full: {cycle_id}")
        active_for_cycle = int(
            self._exec(
                "SELECT COUNT(*) FROM events WHERE cycle_id=? AND status='active'",
                (cycle_id,),
            ).fetchone()[0]
        )
        if active_for_cycle > 0:
            raise ValueError(f"cycle already has active event: {cycle_id}")
        return cycle

    def _insert_event_locked(
        self,
        *,
        event_id: str,
        event: EventCreateInput,
        created_at: datetime,
        expires_at: datetime,
        eval_key: str,
        now_iso: str,
    ) -> None:
        params_snapshot = self._event_params_snapshot_locked(event)
        self._exec(
            """
            INSERT INTO events (
                event_id, asset, horizon, created_at, expires_at, price_t0, pred_low, pred_mid, pred_high,
                payload_json, model_id, model_meta_json, price_source, status, actual_price, current_price,
                metrics_json, params_json, eval_key, note, cycle_id, cycle_seq, cycle_total, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _json_dump(params_snapshot),
                eval_key,
                event.note,
                event.cycle_id,
                event.cycle_seq,
                event.cycle_total,
                now_iso,
            ),
        )
        self._exec(
            "INSERT INTO event_price_samples (event_id, ts, price) VALUES (?, ?, ?)",
            (event_id, _to_iso_utc(created_at), float(event.price_t0)),
        )

    def create_events_batch_for_cycle(
        self,
        cycle_id: str,
        cycle_seq: int,
        cycle_total: int,
        events: list[EventCreateInput],
    ) -> list[dict[str, Any]]:
        if not events:
            return []
        event_ids: list[str] = []
        created_at = events[0].created_at.astimezone(timezone.utc)
        eval_key = _build_event_eval_key(events[0].asset, events[0].horizon, created_at)
        now_iso = _to_iso_utc(_utc_now())
        with self._lock:
            try:
                self._validate_cycle_for_new_run_locked(cycle_id)
                for event in events:
                    if str(event.cycle_id) != cycle_id:
                        raise ValueError("all batch events must reference the same cycle_id")
                    if int(event.cycle_seq or 0) != int(cycle_seq):
                        raise ValueError("all batch events must share cycle_seq")
                    if int(event.cycle_total or 0) != int(cycle_total):
                        raise ValueError("all batch events must share cycle_total")
                    event_created_at = event.created_at.astimezone(timezone.utc)
                    if event_created_at != created_at:
                        raise ValueError("tournament batch events must share the same created_at")
                    event_eval_key = _build_event_eval_key(event.asset, event.horizon, event_created_at)
                    if event_eval_key != eval_key:
                        raise ValueError("tournament batch events must share the same eval_key")
                    event_id = str(uuid.uuid4())
                    expires_at = event_created_at + timedelta(seconds=HORIZON_TO_SECONDS[event.horizon])
                    self._insert_event_locked(
                        event_id=event_id,
                        event=event,
                        created_at=event_created_at,
                        expires_at=expires_at,
                        eval_key=event_eval_key,
                        now_iso=now_iso,
                    )
                    event_ids.append(event_id)
                self._exec(
                    """
                    UPDATE cycles
                    SET launched_runs = launched_runs + 1,
                        last_event_id = ?,
                        updated_at = ?
                    WHERE cycle_id = ? AND status='running'
                    """,
                    (event_ids[0], now_iso, cycle_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return [self.get_event(event_id) for event_id in event_ids]

    def _finalize_tournament_run_if_ready_locked(self, cycle_id: str, cycle_seq: int, now_iso: str) -> None:
        cycle_row = self._exec("SELECT * FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
        if cycle_row is None:
            return
        cycle = self._cycle_from_row(cycle_row)
        accounted_runs = int(cycle.get("completed_runs", 0)) + int(cycle.get("cancelled_runs", 0))
        target_seq = int(cycle_seq)
        if target_seq <= 0 or target_seq > int(cycle.get("launched_runs", 0)) or target_seq <= accounted_runs:
            return
        active_count = int(
            self._exec(
                "SELECT COUNT(*) FROM events WHERE cycle_id=? AND cycle_seq=? AND status='active'",
                (cycle_id, target_seq),
            ).fetchone()[0]
        )
        if active_count > 0:
            return
        rows = self._exec(
            "SELECT status FROM events WHERE cycle_id=? AND cycle_seq=? ORDER BY event_id ASC",
            (cycle_id, target_seq),
        ).fetchall()
        if not rows:
            return
        next_column = "cancelled_runs" if any(str(row["status"]) == "cancelled" for row in rows) else "completed_runs"
        self._exec(
            f"""
            UPDATE cycles
            SET {next_column} = {next_column} + 1,
                updated_at = ?
            WHERE cycle_id = ?
            """,
            (now_iso, cycle_id),
        )
        self._exec(
            """
            UPDATE cycles
            SET status='completed',
                updated_at=?
            WHERE cycle_id=?
              AND launched_runs >= total_runs
              AND (completed_runs + cancelled_runs) >= total_runs
              AND status='running'
            """,
            (now_iso, cycle_id),
        )

    def create_event(self, event: EventCreateInput) -> dict[str, Any]:
        if event.horizon not in HORIZON_TO_SECONDS:
            raise ValueError(f"unsupported horizon: {event.horizon}")
        event_id = str(uuid.uuid4())
        created_at = event.created_at.astimezone(timezone.utc)
        expires_at = created_at + timedelta(seconds=HORIZON_TO_SECONDS[event.horizon])
        eval_key = _build_event_eval_key(event.asset, event.horizon, created_at)
        now_iso = _to_iso_utc(_utc_now())

        with self._lock:
            try:
                if event.cycle_id:
                    cycle = self._validate_cycle_for_new_run_locked(str(event.cycle_id))
                    if str(cycle.get("mode", "single")) == "tournament":
                        raise ValueError(f"tournament cycle requires batch event creation: {event.cycle_id}")
                self._insert_event_locked(
                    event_id=event_id,
                    event=event,
                    created_at=created_at,
                    expires_at=expires_at,
                    eval_key=eval_key,
                    now_iso=now_iso,
                )
                if event.cycle_id:
                    self._exec(
                        """
                        UPDATE cycles
                        SET launched_runs = launched_runs + 1,
                            last_event_id = ?,
                            updated_at = ?
                        WHERE cycle_id = ? AND status='running'
                        """,
                        (event_id, now_iso, event.cycle_id),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
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
        cycle_id: str | None = None,
        model_id: str | None = None,
        model_ids: list[str] | None = None,
        exclude_stale: bool = False,
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
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        selected_model_ids: list[str] = []
        if model_id:
            selected_model_ids.append(str(model_id))
        if model_ids:
            for raw_model_id in model_ids:
                cleaned_model_id = str(raw_model_id).strip()
                if cleaned_model_id and cleaned_model_id not in selected_model_ids:
                    selected_model_ids.append(cleaned_model_id)
        if selected_model_ids:
            if len(selected_model_ids) == 1:
                clauses.append("model_id = ?")
                params.append(selected_model_ids[0])
            else:
                placeholders = ", ".join("?" for _ in selected_model_ids)
                clauses.append(f"model_id IN ({placeholders})")
                params.extend(selected_model_ids)
        if created_from:
            clauses.append("created_at >= ?")
            params.append(created_from)
        if created_to:
            clauses.append("created_at <= ?")
            params.append(created_to)
        if bool(exclude_stale):
            clauses.append("payload_json NOT LIKE '%\"stale_data\": true%'")
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
            "cycle_seq": "cycle_seq",
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
        now_iso = _to_iso_utc(completed_at)
        with self._lock:
            row = self._exec(
                "SELECT cycle_id, cycle_seq FROM events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            updated = self._exec(
                """
                UPDATE events
                SET status='completed',
                    actual_price=?,
                    current_price=?,
                    metrics_json=?,
                    updated_at=?
                WHERE event_id=? AND status='active'
                """,
                (
                    float(actual_price),
                    float(actual_price),
                    _json_dump(metrics),
                    now_iso,
                    event_id,
                ),
            )
            if updated.rowcount > 0 and row and row["cycle_id"]:
                cycle_id = str(row["cycle_id"])
                cycle_row = self._exec("SELECT mode FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
                cycle_mode = str(cycle_row["mode"]) if cycle_row and cycle_row["mode"] is not None else "single"
                if cycle_mode == "tournament":
                    self._finalize_tournament_run_if_ready_locked(cycle_id, int(row["cycle_seq"] or 0), now_iso)
                else:
                    self._exec(
                        """
                        UPDATE cycles
                        SET completed_runs = completed_runs + 1,
                            updated_at = ?
                        WHERE cycle_id = ?
                        """,
                        (now_iso, cycle_id),
                    )
                    self._exec(
                        """
                        UPDATE cycles
                        SET status='completed',
                            updated_at=?
                        WHERE cycle_id=?
                          AND launched_runs >= total_runs
                          AND (completed_runs + cancelled_runs) >= total_runs
                          AND status='running'
                        """,
                        (now_iso, cycle_id),
                    )
            self._conn.commit()

    def cancel_event(self, event_id: str, note: str | None = None) -> None:
        now = _to_iso_utc(_utc_now())
        metrics = {"cancelled_reason": note or "cancelled_by_user"}
        with self._lock:
            row = self._exec(
                "SELECT cycle_id, cycle_seq FROM events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            updated = self._exec(
                "UPDATE events SET status='cancelled', metrics_json=?, updated_at=? WHERE event_id=? AND status='active'",
                (_json_dump(metrics), now, event_id),
            )
            if updated.rowcount > 0 and row and row["cycle_id"]:
                cycle_id = str(row["cycle_id"])
                cycle_row = self._exec("SELECT mode FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
                cycle_mode = str(cycle_row["mode"]) if cycle_row and cycle_row["mode"] is not None else "single"
                if cycle_mode == "tournament":
                    self._finalize_tournament_run_if_ready_locked(cycle_id, int(row["cycle_seq"] or 0), now)
                else:
                    self._exec(
                        """
                        UPDATE cycles
                        SET cancelled_runs = cancelled_runs + 1,
                            updated_at = ?
                        WHERE cycle_id = ?
                        """,
                        (now, cycle_id),
                    )
                    self._exec(
                        """
                        UPDATE cycles
                        SET status='completed',
                            updated_at=?
                        WHERE cycle_id=?
                          AND launched_runs >= total_runs
                          AND (completed_runs + cancelled_runs) >= total_runs
                          AND status='running'
                        """,
                        (now, cycle_id),
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

    def metrics_summary(
        self,
        *,
        days: int = 30,
        horizon: str | None = None,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        start_ts = _to_iso_utc(_utc_now() - timedelta(days=max(1, int(days))))
        clauses = ["status='completed'", "created_at >= ?"]
        params: list[Any] = [start_ts]
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
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

    def compare_metrics(
        self,
        *,
        days: int = 30,
        asset: str = "BTC",
        horizon: str | None = None,
        cycle_id: str | None = None,
        model_ids: list[str] | None = None,
        mode: str = "simple",
        exclude_stale: bool = False,
        group_limit: int = 200,
    ) -> dict[str, Any]:
        mode_clean = str(mode or "simple").strip().lower()
        if mode_clean not in {"simple", "matched", "grouped"}:
            raise ValueError(f"unsupported compare mode: {mode}")

        start_ts = _to_iso_utc(_utc_now() - timedelta(days=max(1, int(days))))
        clauses = ["status='completed'", "created_at >= ?", "asset = ?"]
        params: list[Any] = [start_ts, str(asset).upper()]
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        if bool(exclude_stale):
            clauses.append("payload_json NOT LIKE '%\"stale_data\": true%'")

        selected_model_ids: list[str] = []
        if model_ids:
            for raw_model_id in model_ids:
                cleaned_model_id = str(raw_model_id).strip()
                if cleaned_model_id and cleaned_model_id not in selected_model_ids:
                    selected_model_ids.append(cleaned_model_id)
        if selected_model_ids:
            placeholders = ", ".join("?" for _ in selected_model_ids)
            clauses.append(f"model_id IN ({placeholders})")
            params.extend(selected_model_ids)

        where = " AND ".join(clauses)
        with self._lock:
            event_rows = self._exec(
                f"""
                SELECT event_id, model_id, asset, created_at, cycle_id, eval_key, horizon, metrics_json
                FROM events
                WHERE {where}
                ORDER BY created_at DESC
                """,
                tuple(params),
            ).fetchall()

            seen_model_ids = {str(row["model_id"]) for row in event_rows if row["model_id"]}
            lookup_model_ids = list(dict.fromkeys([*selected_model_ids, *sorted(seen_model_ids)]))
            model_lookup: dict[str, dict[str, Any]] = {}
            if lookup_model_ids:
                model_placeholders = ", ".join("?" for _ in lookup_model_ids)
                rows = self._exec(
                    f"SELECT * FROM models WHERE model_id IN ({model_placeholders})",
                    tuple(lookup_model_ids),
                ).fetchall()
                model_lookup = {
                    str(row["model_id"]): self._model_from_row(row)
                    for row in rows
                }

        if not event_rows:
            return {
                "mode": mode_clean,
                "asset": str(asset).upper(),
                "horizon": horizon,
                "cycle_id": cycle_id,
                "period_days": int(days),
                "total_events": 0,
                "total_completed": 0,
                "total_groups": 0,
                "compared_model_ids": selected_model_ids,
                "items": [
                    {
                        "model_id": model_id_value,
                        "model_name": (model_lookup.get(model_id_value) or {}).get("name"),
                        "asset": (model_lookup.get(model_id_value) or {}).get("asset", str(asset).upper()),
                        "horizon": (model_lookup.get(model_id_value) or {}).get("horizon"),
                        **self._compare_model_fields(model_lookup.get(model_id_value) or {}),
                        "status": (model_lookup.get(model_id_value) or {}).get("status"),
                        "is_production": bool((model_lookup.get(model_id_value) or {}).get("is_production", False)),
                        "total_events": 0,
                        "completed_events": 0,
                        "matched_groups": 0,
                        "group_wins": 0,
                        "coverage": None,
                        "avg_width_pct": None,
                        "avg_abs_error": None,
                        "avg_rel_error": None,
                        "avg_interval_score": None,
                        "avg_wis": None,
                    }
                    for model_id_value in selected_model_ids
                ],
                "groups": [],
            }

        parsed_rows: list[dict[str, Any]] = []
        for row in event_rows:
            metrics = _json_load(row["metrics_json"], {})
            parsed_rows.append(
                {
                    "event_id": str(row["event_id"]),
                    "model_id": str(row["model_id"]),
                    "asset": str(row["asset"]),
                    "created_at": str(row["created_at"]),
                    "cycle_id": str(row["cycle_id"] or "") or None,
                    "eval_key": str(row["eval_key"] or "") or None,
                    "horizon": str(row["horizon"]),
                    "hit": metrics.get("hit") if isinstance(metrics, dict) else None,
                    "abs_error": None if not isinstance(metrics, dict) else metrics.get("abs_error"),
                    "rel_error": None if not isinstance(metrics, dict) else metrics.get("rel_error"),
                    "width_pct": None if not isinstance(metrics, dict) else metrics.get("width_pct"),
                    "interval_score": None if not isinstance(metrics, dict) else metrics.get("interval_score"),
                    "wis": None if not isinstance(metrics, dict) else metrics.get("wis"),
                }
            )

        eligible_rows = parsed_rows
        eligible_groups: dict[str, list[dict[str, Any]]] = {}
        if mode_clean == "matched":
            eligible_rows, eligible_groups = self._matched_compare_groups(parsed_rows, selected_model_ids)
        elif mode_clean == "grouped":
            grouped_rows: dict[str, list[dict[str, Any]]] = {}
            for row in parsed_rows:
                group_key = f"{row['asset']}|{row['horizon']}"
                grouped_rows.setdefault(group_key, []).append(row)
            eligible_groups = grouped_rows
            eligible_rows = [item for group_rows in eligible_groups.values() for item in group_rows]

        stats: dict[str, dict[str, Any]] = {}
        group_presence: dict[str, set[str]] = {}
        group_wins: dict[str, int] = {}

        if mode_clean in {"matched", "grouped"}:
            for group_key, group_rows in eligible_groups.items():
                participating_models = sorted({str(item["model_id"]) for item in group_rows})
                for model_id_value in participating_models:
                    group_presence.setdefault(model_id_value, set()).add(group_key)
                ranked_rows = [item for item in group_rows if isinstance(item.get("hit"), bool) and item.get("hit")]
                if not ranked_rows:
                    ranked_rows = list(group_rows)
                with_error = [item for item in ranked_rows if item.get("abs_error") is not None]
                if with_error:
                    ranked_rows = with_error
                winner = min(
                    ranked_rows,
                    key=lambda item: (
                        float(item["abs_error"]) if item.get("abs_error") is not None else float("inf"),
                        item["created_at"],
                        item["model_id"],
                    ),
                )
                group_wins[winner["model_id"]] = group_wins.get(winner["model_id"], 0) + 1

        for row in eligible_rows:
            model_id_value = str(row["model_id"])
            model_meta = model_lookup.get(model_id_value, {})
            bucket = stats.setdefault(
                model_id_value,
                {
                    "model_id": model_id_value,
                    "model_name": model_meta.get("name"),
                    "asset": model_meta.get("asset", row.get("asset")),
                    "horizon": model_meta.get("horizon", row.get("horizon")),
                    **self._compare_model_fields(model_meta),
                    "status": model_meta.get("status"),
                    "is_production": bool(model_meta.get("is_production", False)),
                    "total_events": 0,
                    "completed_events": 0,
                    "matched_groups": 0,
                    "group_wins": 0,
                    "_hits": 0,
                    "_width_sum": 0.0,
                    "_width_count": 0,
                    "_abs_sum": 0.0,
                    "_abs_count": 0,
                    "_rel_sum": 0.0,
                    "_rel_count": 0,
                    "_interval_score_sum": 0.0,
                    "_interval_score_count": 0,
                    "_wis_sum": 0.0,
                    "_wis_count": 0,
                },
            )
            bucket["total_events"] += 1
            bucket["completed_events"] += 1
            if isinstance(row.get("hit"), bool) and row["hit"]:
                bucket["_hits"] += 1
            if row.get("width_pct") is not None:
                bucket["_width_sum"] += float(row["width_pct"])
                bucket["_width_count"] += 1
            if row.get("abs_error") is not None:
                bucket["_abs_sum"] += float(row["abs_error"])
                bucket["_abs_count"] += 1
            if row.get("rel_error") is not None:
                bucket["_rel_sum"] += float(row["rel_error"])
                bucket["_rel_count"] += 1
            if row.get("interval_score") is not None:
                bucket["_interval_score_sum"] += float(row["interval_score"])
                bucket["_interval_score_count"] += 1
            if row.get("wis") is not None:
                bucket["_wis_sum"] += float(row["wis"])
                bucket["_wis_count"] += 1

        ordered_model_ids = selected_model_ids or list(dict.fromkeys([str(item["model_id"]) for item in eligible_rows]))
        for model_id_value in ordered_model_ids:
            model_meta = model_lookup.get(model_id_value, {})
            bucket = stats.setdefault(
                model_id_value,
                {
                    "model_id": model_id_value,
                    "model_name": model_meta.get("name"),
                    "asset": model_meta.get("asset", str(asset).upper()),
                    "horizon": model_meta.get("horizon", horizon),
                    **self._compare_model_fields(model_meta),
                    "status": model_meta.get("status"),
                    "is_production": bool(model_meta.get("is_production", False)),
                    "total_events": 0,
                    "completed_events": 0,
                    "matched_groups": 0,
                    "group_wins": 0,
                    "_hits": 0,
                    "_width_sum": 0.0,
                    "_width_count": 0,
                    "_abs_sum": 0.0,
                    "_abs_count": 0,
                    "_rel_sum": 0.0,
                    "_rel_count": 0,
                    "_interval_score_sum": 0.0,
                    "_interval_score_count": 0,
                    "_wis_sum": 0.0,
                    "_wis_count": 0,
                },
            )
            bucket["matched_groups"] = len(group_presence.get(model_id_value, set()))
            bucket["group_wins"] = int(group_wins.get(model_id_value, 0))

        items: list[dict[str, Any]] = []
        for model_id_value in ordered_model_ids:
            bucket = stats[model_id_value]
            completed_count = int(bucket["completed_events"])
            width_count = int(bucket["_width_count"])
            abs_count = int(bucket["_abs_count"])
            rel_count = int(bucket["_rel_count"])
            interval_score_count = int(bucket["_interval_score_count"])
            wis_count = int(bucket["_wis_count"])
            items.append(
                {
                    "model_id": model_id_value,
                    "model_name": bucket.get("model_name"),
                    "asset": bucket.get("asset"),
                    "horizon": bucket.get("horizon"),
                    "base_timeframe": bucket.get("base_timeframe"),
                    "steps_ahead": bucket.get("steps_ahead"),
                    "quantiles_count": bucket.get("quantiles_count"),
                    "target_coverage_effective": bucket.get("target_coverage_effective"),
                    "status": bucket.get("status"),
                    "is_production": bool(bucket.get("is_production", False)),
                    "total_events": int(bucket["total_events"]),
                    "completed_events": completed_count,
                    "matched_groups": int(bucket.get("matched_groups", 0)),
                    "group_wins": int(bucket.get("group_wins", 0)),
                    "coverage": (bucket["_hits"] / completed_count) if completed_count else None,
                    "avg_width_pct": (bucket["_width_sum"] / width_count) if width_count else None,
                    "avg_abs_error": (bucket["_abs_sum"] / abs_count) if abs_count else None,
                    "avg_rel_error": (bucket["_rel_sum"] / rel_count) if rel_count else None,
                    "avg_interval_score": (bucket["_interval_score_sum"] / interval_score_count) if interval_score_count else None,
                    "avg_wis": (bucket["_wis_sum"] / wis_count) if wis_count else None,
                }
            )

        if not selected_model_ids:
            ordered_model_ids = [str(item["model_id"]) for item in items]

        groups: list[dict[str, Any]] = []
        if mode_clean in {"matched", "grouped"}:
            sorted_groups = sorted(
                eligible_groups.items(),
                key=lambda pair: max(item["created_at"] for item in pair[1]),
                reverse=True,
            )
            for group_key, group_rows in sorted_groups[: max(1, min(1000, int(group_limit)))]:
                hit_models = sorted(
                    str(item["model_id"])
                    for item in group_rows
                    if isinstance(item.get("hit"), bool) and item["hit"]
                )
                ranked_rows = [item for item in group_rows if isinstance(item.get("hit"), bool) and item.get("hit")]
                if not ranked_rows:
                    ranked_rows = list(group_rows)
                with_error = [item for item in ranked_rows if item.get("abs_error") is not None]
                if with_error:
                    ranked_rows = with_error
                winner = min(
                    ranked_rows,
                    key=lambda item: (
                        float(item["abs_error"]) if item.get("abs_error") is not None else float("inf"),
                        item["created_at"],
                        item["model_id"],
                    ),
                )
                groups.append(
                    {
                        "key": group_key,
                        "created_at": max(item["created_at"] for item in group_rows),
                        "cycle_id": next((item["cycle_id"] for item in group_rows if item.get("cycle_id")), None),
                        "event_count": len(group_rows),
                        "model_ids": sorted({str(item["model_id"]) for item in group_rows}),
                        "best_model_id": str(winner["model_id"]),
                        "best_abs_error": None if winner.get("abs_error") is None else float(winner["abs_error"]),
                        "hit_models": hit_models,
                    }
                )

        items.sort(
            key=lambda item: (
                1 if item["coverage"] is None else 0,
                -(float(item["coverage"]) if item["coverage"] is not None else -1.0),
                float(item["avg_abs_error"]) if item["avg_abs_error"] is not None else float("inf"),
                item["model_id"],
            )
        )

        return {
            "mode": mode_clean,
            "asset": str(asset).upper(),
            "horizon": horizon,
            "cycle_id": cycle_id,
            "period_days": int(days),
            "total_events": len(eligible_rows),
            "total_completed": len(eligible_rows),
            "total_groups": len(eligible_groups),
            "compared_model_ids": ordered_model_ids,
            "items": items,
            "groups": groups,
        }

    def _load_compare_rows(
        self,
        *,
        days: int,
        asset: str | None = None,
        horizon: str | None = None,
        cycle_id: str | None = None,
        model_ids: list[str] | None = None,
        exclude_stale: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
        start_ts = _to_iso_utc(_utc_now() - timedelta(days=max(1, int(days))))
        clauses = ["status='completed'", "created_at >= ?"]
        params: list[Any] = [start_ts]
        asset_clean = str(asset or "").strip().upper()
        if asset_clean:
            clauses.append("asset = ?")
            params.append(asset_clean)
        if horizon:
            clauses.append("horizon = ?")
            params.append(horizon)
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)

        selected_model_ids: list[str] = []
        if model_ids:
            for raw_model_id in model_ids:
                cleaned_model_id = str(raw_model_id).strip()
                if cleaned_model_id and cleaned_model_id not in selected_model_ids:
                    selected_model_ids.append(cleaned_model_id)
        if selected_model_ids:
            placeholders = ", ".join("?" for _ in selected_model_ids)
            clauses.append(f"model_id IN ({placeholders})")
            params.extend(selected_model_ids)
        if bool(exclude_stale):
            clauses.append("payload_json NOT LIKE '%\"stale_data\": true%'")

        where = " AND ".join(clauses)
        with self._lock:
            event_rows = self._exec(
                f"""
                SELECT event_id, model_id, asset, horizon, created_at, cycle_id, eval_key, metrics_json
                FROM events
                WHERE {where}
                ORDER BY created_at DESC
                """,
                tuple(params),
            ).fetchall()
            seen_model_ids = {str(row["model_id"]) for row in event_rows if row["model_id"]}
            lookup_model_ids = list(dict.fromkeys([*selected_model_ids, *sorted(seen_model_ids)]))
            model_lookup: dict[str, dict[str, Any]] = {}
            if lookup_model_ids:
                model_placeholders = ", ".join("?" for _ in lookup_model_ids)
                rows = self._exec(
                    f"SELECT * FROM models WHERE model_id IN ({model_placeholders})",
                    tuple(lookup_model_ids),
                ).fetchall()
                model_lookup = {
                    str(row["model_id"]): self._model_from_row(row)
                    for row in rows
                }

        parsed_rows: list[dict[str, Any]] = []
        for row in event_rows:
            metrics = _json_load(row["metrics_json"], {})
            parsed_rows.append(
                {
                    "event_id": str(row["event_id"]),
                    "model_id": str(row["model_id"]),
                    "asset": str(row["asset"]),
                    "horizon": str(row["horizon"]),
                    "created_at": str(row["created_at"]),
                    "cycle_id": str(row["cycle_id"] or "") or None,
                    "eval_key": str(row["eval_key"] or "") or None,
                    "hit": metrics.get("hit") if isinstance(metrics, dict) else None,
                    "abs_error": None if not isinstance(metrics, dict) else metrics.get("abs_error"),
                    "rel_error": None if not isinstance(metrics, dict) else metrics.get("rel_error"),
                    "width_pct": None if not isinstance(metrics, dict) else metrics.get("width_pct"),
                    "interval_score": None if not isinstance(metrics, dict) else metrics.get("interval_score"),
                    "wis": None if not isinstance(metrics, dict) else metrics.get("wis"),
                }
            )
        return parsed_rows, model_lookup, selected_model_ids

    @staticmethod
    def _compare_model_fields(model_meta: dict[str, Any] | None) -> dict[str, Any]:
        params = {}
        if isinstance(model_meta, dict):
            raw_params = model_meta.get("params_json")
            if isinstance(raw_params, dict):
                params = raw_params

        quantiles_count: int | None = None
        raw_quantiles = params.get("quantiles")
        if isinstance(raw_quantiles, list):
            quantiles_count = len(raw_quantiles)

        base_timeframe = str(params.get("base_timeframe") or "").strip() or None
        steps_ahead = params.get("steps_ahead")
        try:
            steps_ahead = None if steps_ahead is None else int(steps_ahead)
        except (TypeError, ValueError):
            steps_ahead = None

        target_coverage_effective = params.get("target_coverage_effective")
        try:
            target_coverage_effective = None if target_coverage_effective is None else float(target_coverage_effective)
        except (TypeError, ValueError):
            target_coverage_effective = None

        return {
            "base_timeframe": base_timeframe,
            "steps_ahead": steps_ahead,
            "quantiles_count": quantiles_count,
            "target_coverage_effective": target_coverage_effective,
        }

    def _matched_compare_groups(
        self,
        rows: list[dict[str, Any]],
        selected_model_ids: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group_key = str(row.get("eval_key") or "").strip()
            if not group_key:
                continue
            grouped_rows.setdefault(group_key, []).append(row)
        required_models = {item for item in selected_model_ids if item}
        eligible_groups: dict[str, list[dict[str, Any]]] = {}
        for group_key, group_rows in grouped_rows.items():
            present_models = {str(item["model_id"]) for item in group_rows}
            if required_models:
                if required_models.issubset(present_models):
                    eligible_groups[group_key] = group_rows
            elif len(present_models) >= 2:
                eligible_groups[group_key] = group_rows
        eligible_rows = [item for group_rows in eligible_groups.values() for item in group_rows]
        return eligible_rows, eligible_groups

    @staticmethod
    def _rank_compare_group(group_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not group_rows:
            return None
        ranked_rows = [item for item in group_rows if isinstance(item.get("hit"), bool) and item.get("hit")]
        if not ranked_rows:
            ranked_rows = list(group_rows)
        with_error = [item for item in ranked_rows if item.get("abs_error") is not None]
        if with_error:
            ranked_rows = with_error
        return min(
            ranked_rows,
            key=lambda item: (
                float(item["abs_error"]) if item.get("abs_error") is not None else float("inf"),
                item["created_at"],
                item["model_id"],
            ),
        )

    @staticmethod
    def _compare_metric_value(row: dict[str, Any], metric: str) -> float | None:
        metric_clean = str(metric or "coverage").strip().lower()
        if metric_clean == "coverage":
            if not isinstance(row.get("hit"), bool):
                return None
            return 1.0 if bool(row["hit"]) else 0.0
        if metric_clean == "avg_width_pct":
            metric_clean = "width_pct"
        if metric_clean == "avg_abs_error":
            metric_clean = "abs_error"
        if metric_clean == "avg_rel_error":
            metric_clean = "rel_error"
        if metric_clean == "avg_interval_score":
            metric_clean = "interval_score"
        if metric_clean == "avg_wis":
            metric_clean = "wis"
        value = row.get(metric_clean)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def compare_scoreboard(
        self,
        *,
        days: int = 30,
        asset: str | None = "BTC",
        horizon: str | None = None,
        cycle_id: str | None = None,
        model_ids: list[str] | None = None,
        mode: str = "simple",
        exclude_stale: bool = False,
        baseline_model_id: str | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        mode_clean = str(mode or "simple").strip().lower()
        if mode_clean not in {"simple", "matched", "grouped"}:
            raise ValueError(f"unsupported compare mode: {mode}")

        rows, model_lookup, selected_model_ids = self._load_compare_rows(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=model_ids,
            exclude_stale=exclude_stale,
        )
        default_columns = [
            "model_id",
            "asset",
            "horizon",
            "completed_events",
            "coverage",
            "avg_width_pct",
            "avg_abs_error",
            "avg_rel_error",
            "matched_groups",
            "group_wins",
        ]
        columns_clean = [str(item).strip() for item in (columns or default_columns) if str(item).strip()]
        if not rows:
            return {
                "mode": mode_clean,
                "asset": str(asset).upper() if asset else None,
                "horizon": horizon,
                "cycle_id": cycle_id,
                "period_days": int(days),
                "exclude_stale": bool(exclude_stale),
                "baseline_model_id": baseline_model_id,
                "compared_model_ids": selected_model_ids,
                "columns": columns_clean,
                "total_events": 0,
                "total_completed": 0,
                "total_groups": 0,
                "items": [],
            }

        group_map: dict[str, list[dict[str, Any]]] = {}
        eligible_rows = rows
        if mode_clean == "matched":
            eligible_rows, group_map = self._matched_compare_groups(rows, selected_model_ids)
        elif mode_clean == "grouped":
            for row in rows:
                group_key = f"{row['asset']}|{row['horizon']}"
                group_map.setdefault(group_key, []).append(row)

        if mode_clean == "grouped":
            agg_map: dict[tuple[str, str], dict[str, Any]] = {}
            group_winner_ids: dict[str, str] = {}
            for group_key, group_rows in group_map.items():
                per_group: dict[str, dict[str, Any]] = {}
                for row in group_rows:
                    model_id_value = str(row["model_id"])
                    slot = per_group.setdefault(model_id_value, {"sum": 0.0, "count": 0})
                    if row.get("abs_error") is not None:
                        slot["sum"] += float(row["abs_error"])
                        slot["count"] += 1
                ranked = [
                    (model_id_value, (slot["sum"] / slot["count"]) if slot["count"] else float("inf"))
                    for model_id_value, slot in per_group.items()
                ]
                if ranked:
                    ranked.sort(key=lambda item: (item[1], item[0]))
                    group_winner_ids[group_key] = ranked[0][0]

            for row in rows:
                group_key = f"{row['asset']}|{row['horizon']}"
                map_key = (group_key, str(row["model_id"]))
                meta = model_lookup.get(str(row["model_id"]), {})
                bucket = agg_map.setdefault(
                    map_key,
                    {
                        "model_id": str(row["model_id"]),
                        "model_name": meta.get("name"),
                        "asset": row.get("asset"),
                        "horizon": row.get("horizon"),
                        **self._compare_model_fields(meta),
                        "group_key": group_key,
                        "status": meta.get("status"),
                        "is_production": bool(meta.get("is_production", False)),
                        "total_events": 0,
                        "completed_events": 0,
                        "matched_groups": 0,
                        "group_wins": 0,
                        "_hits": 0,
                        "_width_sum": 0.0,
                        "_width_count": 0,
                        "_abs_sum": 0.0,
                        "_abs_count": 0,
                        "_rel_sum": 0.0,
                        "_rel_count": 0,
                        "_interval_score_sum": 0.0,
                        "_interval_score_count": 0,
                        "_wis_sum": 0.0,
                        "_wis_count": 0,
                    },
                )
                bucket["total_events"] += 1
                bucket["completed_events"] += 1
                bucket["matched_groups"] = 1
                if group_winner_ids.get(group_key) == str(row["model_id"]):
                    bucket["group_wins"] = 1
                if isinstance(row.get("hit"), bool) and row["hit"]:
                    bucket["_hits"] += 1
                if row.get("width_pct") is not None:
                    bucket["_width_sum"] += float(row["width_pct"])
                    bucket["_width_count"] += 1
                if row.get("abs_error") is not None:
                    bucket["_abs_sum"] += float(row["abs_error"])
                    bucket["_abs_count"] += 1
                if row.get("rel_error") is not None:
                    bucket["_rel_sum"] += float(row["rel_error"])
                    bucket["_rel_count"] += 1
                if row.get("interval_score") is not None:
                    bucket["_interval_score_sum"] += float(row["interval_score"])
                    bucket["_interval_score_count"] += 1
                if row.get("wis") is not None:
                    bucket["_wis_sum"] += float(row["wis"])
                    bucket["_wis_count"] += 1
            items = list(agg_map.values())
            total_groups = len(group_map)
        else:
            source_rows = eligible_rows
            group_presence: dict[str, set[str]] = {}
            group_wins: dict[str, int] = {}
            if mode_clean == "matched":
                for group_key, group_rows in group_map.items():
                    for model_id_value in {str(item["model_id"]) for item in group_rows}:
                        group_presence.setdefault(model_id_value, set()).add(group_key)
                    winner = self._rank_compare_group(group_rows)
                    if winner is not None:
                        winner_id = str(winner["model_id"])
                        group_wins[winner_id] = group_wins.get(winner_id, 0) + 1

            agg_map: dict[str, dict[str, Any]] = {}
            for row in source_rows:
                model_id_value = str(row["model_id"])
                meta = model_lookup.get(model_id_value, {})
                bucket = agg_map.setdefault(
                    model_id_value,
                    {
                        "model_id": model_id_value,
                        "model_name": meta.get("name"),
                        "asset": meta.get("asset", row.get("asset")),
                        "horizon": meta.get("horizon", row.get("horizon")),
                        **self._compare_model_fields(meta),
                        "group_key": None,
                        "status": meta.get("status"),
                        "is_production": bool(meta.get("is_production", False)),
                        "total_events": 0,
                        "completed_events": 0,
                        "matched_groups": 0,
                        "group_wins": 0,
                        "_hits": 0,
                        "_width_sum": 0.0,
                        "_width_count": 0,
                        "_abs_sum": 0.0,
                        "_abs_count": 0,
                        "_rel_sum": 0.0,
                        "_rel_count": 0,
                        "_interval_score_sum": 0.0,
                        "_interval_score_count": 0,
                        "_wis_sum": 0.0,
                        "_wis_count": 0,
                    },
                )
                bucket["total_events"] += 1
                bucket["completed_events"] += 1
                if mode_clean == "matched":
                    bucket["matched_groups"] = len(group_presence.get(model_id_value, set()))
                    bucket["group_wins"] = int(group_wins.get(model_id_value, 0))
                if isinstance(row.get("hit"), bool) and row["hit"]:
                    bucket["_hits"] += 1
                if row.get("width_pct") is not None:
                    bucket["_width_sum"] += float(row["width_pct"])
                    bucket["_width_count"] += 1
                if row.get("abs_error") is not None:
                    bucket["_abs_sum"] += float(row["abs_error"])
                    bucket["_abs_count"] += 1
                if row.get("rel_error") is not None:
                    bucket["_rel_sum"] += float(row["rel_error"])
                    bucket["_rel_count"] += 1
                if row.get("interval_score") is not None:
                    bucket["_interval_score_sum"] += float(row["interval_score"])
                    bucket["_interval_score_count"] += 1
                if row.get("wis") is not None:
                    bucket["_wis_sum"] += float(row["wis"])
                    bucket["_wis_count"] += 1
            items = list(agg_map.values())
            total_groups = len(group_map)

        for item in items:
            count = max(0, int(item["completed_events"]))
            item["coverage"] = (item["_hits"] / count) if count else None
            item["avg_width_pct"] = (item["_width_sum"] / item["_width_count"]) if item["_width_count"] else None
            item["avg_abs_error"] = (item["_abs_sum"] / item["_abs_count"]) if item["_abs_count"] else None
            item["avg_rel_error"] = (item["_rel_sum"] / item["_rel_count"]) if item["_rel_count"] else None
            item["avg_interval_score"] = (
                (item["_interval_score_sum"] / item["_interval_score_count"])
                if item["_interval_score_count"]
                else None
            )
            item["avg_wis"] = (item["_wis_sum"] / item["_wis_count"]) if item["_wis_count"] else None
            del item["_hits"]
            del item["_width_sum"]
            del item["_width_count"]
            del item["_abs_sum"]
            del item["_abs_count"]
            del item["_rel_sum"]
            del item["_rel_count"]
            del item["_interval_score_sum"]
            del item["_interval_score_count"]
            del item["_wis_sum"]
            del item["_wis_count"]
            item["delta_coverage"] = None
            item["delta_avg_width_pct"] = None
            item["delta_avg_abs_error"] = None
            item["delta_avg_rel_error"] = None
            item["delta_avg_interval_score"] = None
            item["delta_avg_wis"] = None

        baseline_clean = str(baseline_model_id or "").strip()
        if baseline_clean:
            baseline_rows = {
                (item.get("group_key"), item["model_id"]): item
                for item in items
                if item["model_id"] == baseline_clean
            }
            for item in items:
                baseline = baseline_rows.get((item.get("group_key"), baseline_clean)) or baseline_rows.get((None, baseline_clean))
                if baseline is None or item["model_id"] == baseline_clean:
                    continue
                for metric_name, delta_name in (
                    ("coverage", "delta_coverage"),
                    ("avg_width_pct", "delta_avg_width_pct"),
                    ("avg_abs_error", "delta_avg_abs_error"),
                    ("avg_rel_error", "delta_avg_rel_error"),
                    ("avg_interval_score", "delta_avg_interval_score"),
                    ("avg_wis", "delta_avg_wis"),
                ):
                    if item.get(metric_name) is None or baseline.get(metric_name) is None:
                        continue
                    item[delta_name] = float(item[metric_name]) - float(baseline[metric_name])

        items.sort(
            key=lambda item: (
                item.get("group_key") or "",
                1 if item["coverage"] is None else 0,
                -(float(item["coverage"]) if item["coverage"] is not None else -1.0),
                float(item["avg_abs_error"]) if item["avg_abs_error"] is not None else float("inf"),
                item["model_id"],
            )
        )
        compared_model_ids = selected_model_ids or list(dict.fromkeys([str(item["model_id"]) for item in items]))
        total_completed = sum(int(item["completed_events"]) for item in items)
        return {
            "mode": mode_clean,
            "asset": str(asset).upper() if asset else None,
            "horizon": horizon,
            "cycle_id": cycle_id,
            "period_days": int(days),
            "exclude_stale": bool(exclude_stale),
            "baseline_model_id": baseline_model_id,
            "compared_model_ids": compared_model_ids,
            "columns": columns_clean,
            "total_events": total_completed,
            "total_completed": total_completed,
            "total_groups": total_groups,
            "items": items,
        }

    def compare_series(
        self,
        *,
        days: int = 30,
        asset: str | None = "BTC",
        horizon: str | None = None,
        cycle_id: str | None = None,
        model_ids: list[str] | None = None,
        mode: str = "simple",
        exclude_stale: bool = False,
        metric: str = "coverage",
        rolling_days: int = 1,
    ) -> dict[str, Any]:
        mode_clean = str(mode or "simple").strip().lower()
        if mode_clean not in {"simple", "matched", "grouped"}:
            raise ValueError(f"unsupported compare mode: {mode}")
        rows, _, selected_model_ids = self._load_compare_rows(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=model_ids,
            exclude_stale=exclude_stale,
        )
        eligible_rows = rows
        if mode_clean == "matched":
            eligible_rows, _ = self._matched_compare_groups(rows, selected_model_ids)

        series_map: dict[str, dict[str, Any]] = {}
        for row in eligible_rows:
            group_key = f"{row['asset']}|{row['horizon']}" if mode_clean == "grouped" else None
            key = f"{group_key}::{row['model_id']}" if group_key else str(row["model_id"])
            bucket = series_map.setdefault(
                key,
                {
                    "key": key,
                    "model_id": str(row["model_id"]),
                    "group_key": group_key,
                    "metric": str(metric),
                    "_daily": {},
                },
            )
            value = self._compare_metric_value(row, metric)
            if value is None:
                continue
            day = str(row["created_at"])[:10]
            daily = bucket["_daily"].setdefault(day, {"sum": 0.0, "count": 0})
            daily["sum"] += float(value)
            daily["count"] += 1

        rolling = max(1, int(rolling_days))
        out_series: list[dict[str, Any]] = []
        for bucket in series_map.values():
            day_keys = sorted(bucket["_daily"].keys())
            raw_points = [
                {
                    "date": day,
                    "value": bucket["_daily"][day]["sum"] / max(1, bucket["_daily"][day]["count"]),
                    "count": int(bucket["_daily"][day]["count"]),
                }
                for day in day_keys
            ]
            points: list[dict[str, Any]] = []
            for idx in range(len(raw_points)):
                window = raw_points[max(0, idx - rolling + 1) : idx + 1]
                total_count = sum(int(item["count"]) for item in window)
                if total_count <= 0:
                    continue
                weighted_value = sum(float(item["value"]) * int(item["count"]) for item in window) / total_count
                points.append({"date": raw_points[idx]["date"], "value": weighted_value, "count": total_count})
            out_series.append(
                {
                    "key": bucket["key"],
                    "model_id": bucket["model_id"],
                    "group_key": bucket["group_key"],
                    "metric": bucket["metric"],
                    "points": points,
                }
            )
        out_series.sort(key=lambda item: item["key"])
        compared_model_ids = selected_model_ids or sorted({str(item["model_id"]) for item in eligible_rows})
        return {
            "mode": mode_clean,
            "metric": str(metric),
            "rolling_days": rolling,
            "compared_model_ids": compared_model_ids,
            "series": out_series,
        }

    def compare_hist(
        self,
        *,
        days: int = 30,
        asset: str | None = "BTC",
        horizon: str | None = None,
        cycle_id: str | None = None,
        model_ids: list[str] | None = None,
        mode: str = "simple",
        exclude_stale: bool = False,
        metric: str = "abs_error",
        bins: int = 10,
    ) -> dict[str, Any]:
        mode_clean = str(mode or "simple").strip().lower()
        if mode_clean not in {"simple", "matched", "grouped"}:
            raise ValueError(f"unsupported compare mode: {mode}")
        rows, _, selected_model_ids = self._load_compare_rows(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=model_ids,
            exclude_stale=exclude_stale,
        )
        eligible_rows = rows
        if mode_clean == "matched":
            eligible_rows, _ = self._matched_compare_groups(rows, selected_model_ids)

        values_map: dict[str, dict[str, Any]] = {}
        all_values: list[float] = []
        for row in eligible_rows:
            value = self._compare_metric_value(row, metric)
            if value is None:
                continue
            group_key = f"{row['asset']}|{row['horizon']}" if mode_clean == "grouped" else None
            key = f"{group_key}::{row['model_id']}" if group_key else str(row["model_id"])
            bucket = values_map.setdefault(
                key,
                {"key": key, "model_id": str(row["model_id"]), "group_key": group_key, "values": []},
            )
            numeric_value = float(value)
            bucket["values"].append(numeric_value)
            all_values.append(numeric_value)

        bins_clean = max(1, min(100, int(bins)))
        if not all_values:
            return {
                "mode": mode_clean,
                "metric": str(metric),
                "bins": bins_clean,
                "compared_model_ids": selected_model_ids,
                "series": [],
            }

        min_value = min(all_values)
        max_value = max(all_values)
        span = max_value - min_value
        width = (span / bins_clean) if span > 0 else 1.0
        out_series: list[dict[str, Any]] = []
        for bucket in values_map.values():
            counts = [0 for _ in range(bins_clean)]
            for value in bucket["values"]:
                idx = bins_clean - 1 if span <= 0 else min(bins_clean - 1, max(0, int((value - min_value) / width)))
                counts[idx] += 1
            out_series.append(
                {
                    "key": bucket["key"],
                    "model_id": bucket["model_id"],
                    "group_key": bucket["group_key"],
                    "bins": [
                        {
                            "bin_start": min_value + width * idx,
                            "bin_end": (min_value + width * (idx + 1)) if idx < bins_clean - 1 else max_value,
                            "count": counts[idx],
                        }
                        for idx in range(bins_clean)
                    ],
                }
            )
        out_series.sort(key=lambda item: item["key"])
        compared_model_ids = selected_model_ids or sorted({str(item["model_id"]) for item in eligible_rows})
        return {
            "mode": mode_clean,
            "metric": str(metric),
            "bins": bins_clean,
            "compared_model_ids": compared_model_ids,
            "series": out_series,
        }

    @staticmethod
    def parse_utc(ts: str) -> datetime:
        return _parse_utc(ts)
