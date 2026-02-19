"""Inference API and Web GUI event-management backend."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from src.config import get_runtime_paths, load_config
from src.models.predict import (
    ModelBundle,
    ReturnQuantiles,
    build_price_prediction,
    load_model_bundle,
    predict_return_quantiles,
)
from src.service.cache import CandleCache
from src.service.event_store import EventCreateInput, EventStore, HORIZON_TO_SECONDS
from src.service.schemas import (
    AlertsResponse,
    CreateEventRequest,
    EventListResponse,
    EventPricesResponse,
    EventRecord,
    HealthResponse,
    MetricsSummaryResponse,
    ModelListResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictResponse,
    ProductionModelsResponse,
    StatusResponse,
)

SUPPORTED_HORIZONS = ("5m", "15m", "1h", "4h", "1d", "1w")

FALLBACK_RETURNS: dict[str, ReturnQuantiles] = {
    "5m": ReturnQuantiles(q10=-0.0020, q50=0.0000, q90=0.0020),
    "15m": ReturnQuantiles(q10=-0.0032, q50=0.0001, q90=0.0036),
    "1h": ReturnQuantiles(q10=-0.0068, q50=0.0003, q90=0.0073),
    "4h": ReturnQuantiles(q10=-0.0130, q50=0.0008, q90=0.0140),
    "1d": ReturnQuantiles(q10=-0.0270, q50=0.0015, q90=0.0300),
    "1w": ReturnQuantiles(q10=-0.0700, q50=0.0040, q90=0.0850),
}

CONFIG = load_config()
SUPPORTED_ASSETS = tuple(
    asset
    for asset, info in CONFIG.get("assets", {}).items()
    if str(info.get("status", "enabled")).lower() == "enabled"
)
PATHS = get_runtime_paths(CONFIG)
CANDLE_CACHE = CandleCache(CONFIG)
EVENT_STORE = EventStore(PATHS.artifacts_root / "db" / "events.sqlite3")

_MODEL_BUNDLES: dict[tuple[str, str], ModelBundle | None] = {}
_MODEL_BUNDLES_BY_ID: dict[tuple[str, str, str], ModelBundle | None] = {}
_MODEL_LOCK = Lock()

_WORKER_STOP = Event()
_WORKER_THREAD: Thread | None = None
_WORKER_LOCK = Lock()


@dataclass
class RequestStats:
    total_requests: int = 0
    predict_requests: int = 0
    errors_total: int = 0
    stale_data_responses: int = 0


@dataclass(frozen=True)
class PredictionContext:
    response: PredictResponse
    bundle: ModelBundle | None
    model_version: str


STATS = RequestStats()
_RATE_LOCK = Lock()
_RATE_BUCKET: dict[tuple[str, int], int] = {}


def _check_supported(asset: str, horizon: str) -> tuple[str, str]:
    asset_u = asset.upper()
    if asset_u not in SUPPORTED_ASSETS:
        raise ValueError(f"unsupported asset: {asset}")
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    return asset_u, horizon


def _model_version(asset: str, horizon: str, bundle: ModelBundle | None) -> str:
    if bundle and bundle.metadata.get("model_version"):
        return str(bundle.metadata["model_version"])
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{asset.lower()}_{horizon}_fallback_q_v1_{date_tag}"


def _load_bundle_cached(asset: str, horizon: str) -> ModelBundle | None:
    key = (asset, horizon)
    with _MODEL_LOCK:
        if key in _MODEL_BUNDLES:
            return _MODEL_BUNDLES[key]
        bundle = load_model_bundle(PATHS.artifacts_root, asset, horizon)
        _MODEL_BUNDLES[key] = bundle
        return bundle


def _load_bundle_by_id_cached(asset: str, horizon: str, model_id: str) -> ModelBundle | None:
    key = (asset, horizon, model_id)
    with _MODEL_LOCK:
        if key in _MODEL_BUNDLES_BY_ID:
            return _MODEL_BUNDLES_BY_ID[key]
        bundle = load_model_bundle(PATHS.artifacts_root, asset, horizon, model_version=model_id)
        _MODEL_BUNDLES_BY_ID[key] = bundle
        return bundle


def _timeframe_for_horizon(horizon: str) -> str:
    horizon_cfg = CONFIG.get("horizons_map", {}).get(horizon)
    if not horizon_cfg:
        return horizon
    return str(horizon_cfg.get("timeframe", horizon))


def _model_meta_from_bundle(bundle: ModelBundle | None, model_id: str) -> dict[str, Any]:
    metadata = bundle.metadata if bundle else {}
    return {
        "model_id": model_id,
        "git_commit": str(metadata.get("git_commit", "unknown")),
        "train_time": str(metadata.get("created_at_utc", "unknown")),
        "dataset_hash": str(metadata.get("dataset_hash", "unknown")),
        "config_hash": str(metadata.get("config_hash", "unknown")),
        "metrics": metadata.get("test_metrics_calibrated", {}),
    }


def _sync_models_registry() -> None:
    models_root = PATHS.artifacts_root / "models"
    if not models_root.exists():
        return
    for asset_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        asset = asset_dir.name.upper()
        for horizon_dir in sorted([p for p in asset_dir.iterdir() if p.is_dir()]):
            horizon = horizon_dir.name
            versions = sorted([p for p in horizon_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
            if not versions:
                continue
            latest_model_id = versions[-1].name
            for model_dir in versions:
                metadata_path = model_dir / "metadata.json"
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except Exception:
                        metadata = {}
                model_id = str(metadata.get("model_version", model_dir.name))
                if model_dir == versions[-1]:
                    latest_model_id = model_id
                EVENT_STORE.upsert_model(
                    {
                        "model_id": model_id,
                        "asset": asset,
                        "horizon": horizon,
                        "name": f"{asset} {horizon}",
                        "created_at": str(metadata.get("created_at_utc", datetime.now(timezone.utc).isoformat())),
                        "git_commit": str(metadata.get("git_commit", "unknown")),
                        "train_time": str(metadata.get("created_at_utc", "unknown")),
                        "dataset_hash": str(metadata.get("dataset_hash", "unknown")),
                        "config_hash": str(metadata.get("config_hash", "unknown")),
                        "metrics_json": metadata.get("test_metrics_calibrated", {}),
                        "is_production": model_dir == versions[-1],
                    }
                )
            EVENT_STORE.set_production_model(asset=asset, horizon=horizon, model_id=latest_model_id)


def _predict_internal_ctx(asset: str, horizon: str, model_id: str | None = None) -> PredictionContext:
    asset_u, horizon_u = _check_supported(asset, horizon)
    timeframe = _timeframe_for_horizon(horizon_u)
    feature_row, spot, as_of_ts, stale = CANDLE_CACHE.latest_feature_row(asset_u, timeframe)

    bundle: ModelBundle | None
    if model_id:
        bundle = _load_bundle_by_id_cached(asset_u, horizon_u, model_id)
        if bundle is None:
            raise ValueError(f"model_id not found for {asset_u}/{horizon_u}: {model_id}")
        resolved_model_version = model_id
    else:
        bundle = _load_bundle_cached(asset_u, horizon_u)
        resolved_model_version = _model_version(asset_u, horizon_u, bundle)

    if bundle is not None:
        quantiles = predict_return_quantiles(bundle, feature_row)
    else:
        quantiles = FALLBACK_RETURNS[horizon_u]

    payload = build_price_prediction(
        asset=asset_u,
        horizon=horizon_u,
        as_of=as_of_ts.to_pydatetime() if hasattr(as_of_ts, "to_pydatetime") else as_of_ts,
        price_spot=spot,
        return_quantiles=quantiles,
        nominal=0.80,
        calibration_score=0.92
        if bundle is None
        else float(bundle.metadata.get("calibrator", {}).get("target_coverage", 0.8)),
        drift_flag=False,
        model_version=resolved_model_version,
        stale_data=stale,
    )
    return PredictionContext(
        response=PredictResponse.model_validate(payload),
        bundle=bundle,
        model_version=resolved_model_version,
    )


def _predict_internal(asset: str, horizon: str) -> PredictResponse:
    return _predict_internal_ctx(asset=asset, horizon=horizon).response


def _build_event_metrics(event: dict[str, Any], actual_price: float, now_utc: datetime) -> dict[str, Any]:
    pred_low = float(event["pred_low"])
    pred_mid = float(event["pred_mid"])
    pred_high = float(event["pred_high"])
    price_t0 = float(event["price_t0"])
    hit = pred_low <= actual_price <= pred_high
    abs_error = abs(actual_price - pred_mid)
    rel_error = abs_error / max(price_t0, 1e-12)
    width_pct = ((pred_high - pred_low) / max(price_t0, 1e-12)) * 100.0
    latency_ms = max(0, int((now_utc - EventStore.parse_utc(event["expires_at"])).total_seconds() * 1000))
    return {
        "hit": hit,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "width_pct": width_pct,
        "latency_ms": latency_ms,
    }


def _worker_tick() -> None:
    active = EVENT_STORE.list_active_events()
    if not active:
        return
    now_utc = datetime.now(timezone.utc)
    for event in active:
        try:
            current_price, sample_ts, stale = CANDLE_CACHE.latest_price(event["asset"], timeframe="1m")
            sample_dt = sample_ts.to_pydatetime() if hasattr(sample_ts, "to_pydatetime") else sample_ts
            EVENT_STORE.add_price_sample(event["event_id"], sample_dt, current_price)
            EVENT_STORE.update_event_live(event["event_id"], current_price=current_price, updated_at=now_utc)

            expires_at = EventStore.parse_utc(event["expires_at"])
            if now_utc >= expires_at:
                metrics = _build_event_metrics(event, actual_price=current_price, now_utc=now_utc)
                metrics["stale_data"] = bool(stale)
                EVENT_STORE.complete_event(
                    event["event_id"],
                    actual_price=current_price,
                    metrics=metrics,
                    completed_at=now_utc,
                )
        except Exception as exc:
            EVENT_STORE.add_alert(
                level="error",
                code="worker_tick_failed",
                message=str(exc),
                context={"event_id": event.get("event_id")},
            )


def _worker_loop() -> None:
    interval = max(1, int(CONFIG.get("service", {}).get("event_worker_interval_seconds", 5)))
    while not _WORKER_STOP.is_set():
        _worker_tick()
        _WORKER_STOP.wait(interval)


def _start_worker() -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD and _WORKER_THREAD.is_alive():
            return
        _WORKER_STOP.clear()
        _WORKER_THREAD = Thread(target=_worker_loop, daemon=True, name="event-worker")
        _WORKER_THREAD.start()


def _stop_worker() -> None:
    with _WORKER_LOCK:
        _WORKER_STOP.set()
        if _WORKER_THREAD and _WORKER_THREAD.is_alive():
            _WORKER_THREAD.join(timeout=5)


def get_health() -> HealthResponse:
    STATS.total_requests += 1
    return HealthResponse(status="ok")


def get_status() -> StatusResponse:
    STATS.total_requests += 1
    model_map: dict[str, str] = {}
    for asset in SUPPORTED_ASSETS:
        for horizon in SUPPORTED_HORIZONS:
            bundle = _load_bundle_cached(asset, horizon)
            model_map[f"{asset}_{horizon}"] = _model_version(asset, horizon, bundle)
    return StatusResponse(
        mode="price_ranges",
        assets=list(SUPPORTED_ASSETS),
        horizons=list(SUPPORTED_HORIZONS),
        models=model_map,
        data_freshness_seconds=int(CONFIG.get("data", {}).get("cache_ttl_seconds", 300)),
    )


def get_predict(asset: str, horizon: str) -> PredictResponse:
    STATS.total_requests += 1
    STATS.predict_requests += 1
    try:
        response = _predict_internal(asset=asset, horizon=horizon)
    except Exception:
        STATS.errors_total += 1
        raise
    if response.stale_data:
        STATS.stale_data_responses += 1
    return response


def post_predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    predictions = [get_predict(req.asset, req.horizon) for req in request.requests]
    return PredictBatchResponse(predictions=predictions)


def metrics_text() -> str:
    return "\n".join(
        [
            "# HELP ulyanai_requests_total Total API requests.",
            "# TYPE ulyanai_requests_total counter",
            f"ulyanai_requests_total {STATS.total_requests}",
            "# HELP ulyanai_predict_requests_total Total /predict requests.",
            "# TYPE ulyanai_predict_requests_total counter",
            f"ulyanai_predict_requests_total {STATS.predict_requests}",
            "# HELP ulyanai_errors_total Total API errors.",
            "# TYPE ulyanai_errors_total counter",
            f"ulyanai_errors_total {STATS.errors_total}",
            "# HELP ulyanai_stale_data_total Number of stale_data responses.",
            "# TYPE ulyanai_stale_data_total counter",
            f"ulyanai_stale_data_total {STATS.stale_data_responses}",
            "",
        ]
    )


def _auth_dep(request: Request) -> None:
    expected = str(CONFIG.get("service", {}).get("api_key", "dev-key"))
    if not expected:
        return
    header_name = str(CONFIG.get("service", {}).get("api_key_header", "X-API-Key"))
    provided = request.headers.get(header_name) or request.query_params.get("api_key")
    if provided != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _rate_limit_dep(request: Request) -> None:
    limit = int(CONFIG.get("service", {}).get("rate_limit_per_min", 120))
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    minute = int(datetime.now(timezone.utc).timestamp() // 60)
    key = (ip, minute)
    with _RATE_LOCK:
        stale_before = minute - 2
        for existing_key in list(_RATE_BUCKET.keys()):
            if existing_key[1] < stale_before:
                _RATE_BUCKET.pop(existing_key, None)
        current = _RATE_BUCKET.get(key, 0) + 1
        _RATE_BUCKET[key] = current
        if current > limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")


app = FastAPI(title="UlyanAI Inference Service", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    _sync_models_registry()
    _start_worker()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    _stop_worker()


@app.get("/", include_in_schema=False)
def root_ui() -> FileResponse:
    ui_path = PATHS.root / "web" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(ui_path, media_type="text/html; charset=utf-8")


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_health()


@app.get("/v1/status", response_model=StatusResponse)
def status(_: None = Depends(_auth_dep), __: None = Depends(_rate_limit_dep)) -> StatusResponse:
    return get_status()


@app.get("/v1/predict", response_model=PredictResponse)
def predict(
    asset: str,
    horizon: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> PredictResponse:
    try:
        return get_predict(asset=asset, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/predict_batch", response_model=PredictBatchResponse)
def predict_batch(
    request: PredictBatchRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> PredictBatchResponse:
    try:
        return post_predict_batch(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_text()


@app.post("/api/events", response_model=EventRecord)
def create_event(
    request: CreateEventRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> EventRecord:
    try:
        ctx = _predict_internal_ctx(request.asset, request.horizon, request.model_id)
        model_meta = _model_meta_from_bundle(ctx.bundle, ctx.model_version)
        event = EVENT_STORE.create_event(
            EventCreateInput(
                asset=ctx.response.asset,
                horizon=ctx.response.horizon,
                created_at=datetime.now(timezone.utc),
                price_t0=float(ctx.response.price_spot),
                prediction=ctx.response.model_dump(),
                pred_low=float(ctx.response.price_range.low),
                pred_mid=float(ctx.response.median_price),
                pred_high=float(ctx.response.price_range.high),
                model_id=ctx.model_version,
                model_meta=model_meta,
                price_source=request.price_source,
                note=request.note,
            )
        )
        return EventRecord.model_validate(event)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        EVENT_STORE.add_alert(
            level="error",
            code="create_event_failed",
            message=str(exc),
            context={"asset": request.asset, "horizon": request.horizon},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/events", response_model=EventListResponse)
def list_events(
    status: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    result: str | None = Query(default=None, pattern="^(hit|miss)?$"),
    model_id: str | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    q: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> EventListResponse:
    payload = EVENT_STORE.list_events(
        status=status,
        horizon=horizon,
        model_id=model_id,
        result=result,
        created_from=created_from,
        created_to=created_to,
        q=q,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    return EventListResponse.model_validate(payload)


@app.get("/api/events/{event_id}", response_model=EventRecord)
def get_event(
    event_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> EventRecord:
    try:
        return EventRecord.model_validate(EVENT_STORE.get_event(event_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/events/{event_id}/prices", response_model=EventPricesResponse)
def get_event_prices(
    event_id: str,
    limit: int = Query(default=500, ge=1, le=10000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> EventPricesResponse:
    try:
        EVENT_STORE.get_event(event_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EventPricesResponse.model_validate(
        {
            "event_id": event_id,
            "samples": EVENT_STORE.list_price_samples(event_id, limit=limit),
        }
    )


@app.post("/api/events/{event_id}/cancel", response_model=EventRecord)
def cancel_event(
    event_id: str,
    note: str | None = Query(default=None),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> EventRecord:
    try:
        current = EVENT_STORE.get_event(event_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if current["status"] != "active":
        raise HTTPException(status_code=400, detail="only active events can be cancelled")
    EVENT_STORE.cancel_event(event_id, note=note)
    return EventRecord.model_validate(EVENT_STORE.get_event(event_id))


@app.get("/api/models", response_model=ModelListResponse)
def list_models(
    asset: str = Query(default="BTC"),
    horizon: str | None = Query(default=None),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelListResponse:
    _sync_models_registry()
    return ModelListResponse.model_validate({"items": EVENT_STORE.list_models(asset=asset, horizon=horizon)})


@app.get("/api/models/production", response_model=ProductionModelsResponse)
def production_models(
    asset: str = Query(default="BTC"),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ProductionModelsResponse:
    _sync_models_registry()
    return ProductionModelsResponse.model_validate({"items": EVENT_STORE.production_models(asset=asset)})


@app.get("/api/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    days: int = Query(default=30, ge=1, le=3650),
    horizon: str | None = Query(default=None),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> MetricsSummaryResponse:
    return MetricsSummaryResponse.model_validate(EVENT_STORE.metrics_summary(days=days, horizon=horizon))


@app.get("/api/alerts", response_model=AlertsResponse)
def alerts(
    limit: int = Query(default=100, ge=1, le=1000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> AlertsResponse:
    return AlertsResponse.model_validate({"items": EVENT_STORE.list_alerts(limit=limit)})


@app.get("/api/stream/events")
async def stream_events(
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> StreamingResponse:
    interval = max(1, int(CONFIG.get("service", {}).get("event_worker_interval_seconds", 5)))

    async def event_generator():
        while True:
            active = EVENT_STORE.list_events(status="active", page=1, page_size=1000)["items"]
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "active": active,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

