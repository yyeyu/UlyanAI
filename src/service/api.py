"""Inference API and Web GUI event-management backend."""

from __future__ import annotations

import atexit
import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import quote

import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from src.config import get_runtime_paths, load_config
from src.models.predict import (
    ModelBundle,
    ReturnQuantiles,
    build_price_prediction,
    load_model_bundle,
    predict_bundle_outputs,
)
from src.reporting.report_builder import generate_experiment_report_artifacts, resolve_report_paths
from src.service.cache import CandleCache
from src.service.event_store import CycleCreateInput, EventCreateInput, EventStore
from src.service.model_builder import (
    TrainingJobCancelled,
    estimate_sweep_variant_count,
    run_training_job,
    run_walk_forward_job,
    validate_builder_payload,
)
from src.service.model_registry import sync_models_registry
from src.service.schemas import (
    AlertsResponse,
    CompareHistogramResponse,
    CompareMetricsResponse,
    CompareScoreboardResponse,
    CompareSeriesResponse,
    CreateCycleRequest,
    CreateEventRequest,
    CycleListResponse,
    CycleRecord,
    EventListResponse,
    EventPricesResponse,
    EventRecord,
    GenerateReportRequest,
    GenerateReportResponse,
    HealthResponse,
    JobListResponse,
    JobLogsResponse,
    JobRecord,
    ModelBuilderValidateRequest,
    ModelBuilderValidateResponse,
    ModelMetadataUpdateRequest,
    MetricsSummaryResponse,
    ModelListResponse,
    ModelRecord,
    PolymarketCryptoEventsResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictResponse,
    ProductionModelsResponse,
    StatusResponse,
    TournamentCycleSummaryResponse,
    TrainingJobBatchResponse,
    TrainingJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobLogsResponse,
    TrainingJobRecord,
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

CONFIG_DIR = os.getenv("ULYANAI_CONFIG_DIR", "configs")
CONFIG_ROOT = os.getenv("ULYANAI_ROOT", ".")
CONFIG = load_config(CONFIG_DIR)
SUPPORTED_ASSETS = tuple(
    asset
    for asset, info in CONFIG.get("assets", {}).items()
    if str(info.get("status", "enabled")).lower() == "enabled"
)
PATHS = get_runtime_paths(CONFIG, root=CONFIG_ROOT)
CANDLE_CACHE = CandleCache(CONFIG, root=CONFIG_ROOT)
EVENT_STORE = EventStore(PATHS.artifacts_root / "db" / "events.sqlite3")
atexit.register(EVENT_STORE.close)

_MODEL_BUNDLES: dict[tuple[str, str], ModelBundle | None] = {}
_MODEL_BUNDLES_BY_ID: dict[tuple[str, str, str], ModelBundle | None] = {}
_MODEL_LOCK = Lock()

_WORKER_STOP = Event()
_WORKER_THREAD: Thread | None = None
_WORKER_LOCK = Lock()

_JOB_WORKER_STOP = Event()
_JOB_WORKER_THREAD: Thread | None = None
_JOB_WORKER_LOCK = Lock()


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

POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
POLYMARKET_CRYPTO_TAG_SLUG = "crypto"
POLYMARKET_EVENTS_PAGE_SIZE = 500
POLYMARKET_EVENTS_MAX_PAGES = 200
POLYMARKET_REQUEST_TIMEOUT_SEC = 15.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _max_models_per_tournament() -> int:
    return max(2, int(CONFIG.get("service", {}).get("max_models_per_tournament", 20)))


def _tournament_orchestrator_rules() -> dict[str, Any]:
    service_cfg = CONFIG.get("service", {})
    if not isinstance(service_cfg, dict):
        return {}
    raw_rules = service_cfg.get("tournament_orchestrator")
    if not isinstance(raw_rules, dict):
        return {}
    return dict(raw_rules)


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


def _production_model_id(asset: str, horizon: str) -> str | None:
    for item in EVENT_STORE.production_models(asset=asset):
        if str(item.get("horizon")) == horizon and str(item.get("status", "active")) == "active":
            return str(item["model_id"])
    return None


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


def _event_params_snapshot(
    *,
    bundle: ModelBundle | None,
    model_id: str,
    asset: str,
    horizon: str,
    price_source: str,
) -> dict[str, Any]:
    try:
        model = EVENT_STORE.get_model(model_id)
        params_json = model.get("params_json", {})
        if isinstance(params_json, dict) and params_json:
            return dict(params_json)
    except KeyError:
        pass
    metadata = bundle.metadata if bundle else {}
    metadata_params = metadata.get("params_json", {})
    if isinstance(metadata_params, dict) and metadata_params:
        return dict(metadata_params)
    return {
        "asset": asset.upper(),
        "horizon": horizon,
        "model_id": model_id,
        "price_source": price_source,
    }


def _required_timeframe_for_model(
    *,
    asset: str,
    horizon: str,
    model_id: str | None,
    bundle: ModelBundle | None = None,
) -> str:
    if model_id:
        try:
            model = EVENT_STORE.get_model(model_id)
            params_json = model.get("params_json", {})
            if isinstance(params_json, dict):
                base_timeframe = str(params_json.get("base_timeframe", "")).strip()
                if base_timeframe:
                    return base_timeframe
        except KeyError:
            pass
    metadata = bundle.metadata if bundle else {}
    metadata_params = metadata.get("params_json", {})
    if isinstance(metadata_params, dict):
        base_timeframe = str(metadata_params.get("base_timeframe", "")).strip()
        if base_timeframe:
            return base_timeframe
    base_timeframe = str(metadata.get("base_timeframe", "")).strip()
    if base_timeframe:
        return base_timeframe
    return _timeframe_for_horizon(horizon)


def _build_prediction_from_snapshot(
    *,
    asset: str,
    horizon: str,
    model_version: str,
    bundle: ModelBundle | None,
    feature_row: dict[str, Any],
    price_spot: float,
    as_of: Any,
    stale_data: bool,
) -> PredictResponse:
    extra_quantiles: dict[str, float] = {}
    if bundle is not None:
        prediction = predict_bundle_outputs(bundle, feature_row)
        quantiles = prediction.primary
        extra_quantiles = prediction.extra_quantiles
    else:
        quantiles = FALLBACK_RETURNS[horizon]

    payload = build_price_prediction(
        asset=asset,
        horizon=horizon,
        as_of=as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else as_of,
        price_spot=price_spot,
        return_quantiles=quantiles,
        nominal=0.80,
        calibration_score=0.92
        if bundle is None
        else float(bundle.metadata.get("calibrator", {}).get("target_coverage", 0.8)),
        drift_flag=False,
        model_version=model_version,
        stale_data=stale_data,
        extra_return_quantiles=extra_quantiles,
    )
    return PredictResponse.model_validate(payload)


def _to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    raw = value.strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _parse_token_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        from_json = _parse_json_list(value)
        if from_json:
            return [str(item) for item in from_json if item is not None]
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def _normalize_polymarket_market(market: dict[str, Any]) -> dict[str, Any]:
    outcomes_raw = _parse_json_list(market.get("outcomes"))
    outcome_prices_raw = _parse_json_list(market.get("outcomePrices"))
    outcome_prices = [price for price in (_to_optional_float(item) for item in outcome_prices_raw) if price is not None]
    token_ids = _parse_token_ids(market.get("clobTokenIds"))
    return {
        "market_id": str(market.get("id")) if market.get("id") is not None else None,
        "condition_id": str(market.get("conditionId")) if market.get("conditionId") is not None else None,
        "slug": str(market.get("slug")) if market.get("slug") is not None else None,
        "question": str(market.get("question")) if market.get("question") is not None else None,
        "active": bool(market["active"]) if "active" in market else None,
        "closed": bool(market["closed"]) if "closed" in market else None,
        "end_date": str(market.get("endDate")) if market.get("endDate") is not None else None,
        "volume": _to_optional_float(market.get("volumeNum", market.get("volume"))),
        "volume_clob": _to_optional_float(market.get("volumeClob")),
        "best_ask": _to_optional_float(market.get("bestAsk")),
        "last_trade_price": _to_optional_float(market.get("lastTradePrice")),
        "outcomes": [str(item) for item in outcomes_raw if item is not None],
        "outcome_prices": outcome_prices,
        "yes_token_id": token_ids[0] if len(token_ids) > 0 else None,
        "no_token_id": token_ids[1] if len(token_ids) > 1 else None,
    }


def _normalize_polymarket_event(event: dict[str, Any]) -> dict[str, Any]:
    markets_raw = event.get("markets")
    markets: list[dict[str, Any]] = []
    markets_count = 0
    if isinstance(markets_raw, list):
        markets_count = len(markets_raw)
        # Keep only the top market details for UI rendering; full nested market payload is large.
        top_market = next((item for item in markets_raw if isinstance(item, dict)), None)
        if top_market is not None:
            markets = [_normalize_polymarket_market(top_market)]
    tags_raw = event.get("tags")
    tags: list[str] = []
    if isinstance(tags_raw, list):
        for item in tags_raw:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("slug")
            if label is None:
                continue
            text = str(label).strip()
            if text:
                tags.append(text)
    liquidity = _to_optional_float(event.get("liquidityClob"))
    if liquidity is None:
        liquidity = _to_optional_float(event.get("liquidity"))
    return {
        "event_id": str(event.get("id")) if event.get("id") is not None else "",
        "slug": str(event.get("slug")) if event.get("slug") is not None else None,
        "title": str(event.get("title")) if event.get("title") is not None else None,
        "description": str(event.get("description")) if event.get("description") is not None else None,
        "active": bool(event["active"]) if "active" in event else None,
        "closed": bool(event["closed"]) if "closed" in event else None,
        "start_date": str(event.get("startDate")) if event.get("startDate") is not None else None,
        "end_date": str(event.get("endDate")) if event.get("endDate") is not None else None,
        "volume": _to_optional_float(event.get("volume")),
        "volume_24hr": _to_optional_float(event.get("volume24hr")),
        "liquidity": liquidity,
        "open_interest": _to_optional_float(event.get("openInterest")),
        "icon": str(event.get("icon")) if event.get("icon") is not None else None,
        "markets_count": markets_count,
        "markets": markets,
        "tags": tags,
    }


def _fetch_polymarket_events(tag_slug: str, active: bool | None, closed: bool | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    for _ in range(POLYMARKET_EVENTS_MAX_PAGES):
        params: dict[str, Any] = {
            "tag_slug": tag_slug,
            "limit": POLYMARKET_EVENTS_PAGE_SIZE,
            "offset": offset,
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        response = requests.get(
            f"{POLYMARKET_GAMMA_BASE_URL}/events",
            params=params,
            timeout=POLYMARKET_REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("unexpected polymarket gamma response for /events")
        if not payload:
            break
        for row in payload:
            if not isinstance(row, dict):
                continue
            event_id = str(row.get("id") or "").strip()
            if event_id and event_id in seen_ids:
                continue
            if event_id:
                seen_ids.add(event_id)
            items.append(row)
        if len(payload) < POLYMARKET_EVENTS_PAGE_SIZE:
            break
        offset += POLYMARKET_EVENTS_PAGE_SIZE
    return items


def _sync_models_registry() -> None:
    sync_models_registry(store=EVENT_STORE, artifacts_root=PATHS.artifacts_root)


def _predict_internal_ctx(asset: str, horizon: str, model_id: str | None = None) -> PredictionContext:
    asset_u, horizon_u = _check_supported(asset, horizon)

    bundle: ModelBundle | None
    if model_id:
        bundle = _load_bundle_by_id_cached(asset_u, horizon_u, model_id)
        if bundle is None:
            raise ValueError(f"model_id not found for {asset_u}/{horizon_u}: {model_id}")
        resolved_model_version = model_id
    else:
        production_model_id = _production_model_id(asset_u, horizon_u)
        if production_model_id:
            bundle = _load_bundle_by_id_cached(asset_u, horizon_u, production_model_id)
            if bundle is None:
                raise ValueError(f"production model artifact not found for {asset_u}/{horizon_u}: {production_model_id}")
            resolved_model_version = production_model_id
        else:
            bundle = _load_bundle_cached(asset_u, horizon_u)
            resolved_model_version = _model_version(asset_u, horizon_u, bundle)

    timeframe = _required_timeframe_for_model(
        asset=asset_u,
        horizon=horizon_u,
        model_id=resolved_model_version if bundle is not None else None,
        bundle=bundle,
    )
    feature_row, feature_spot, feature_as_of_ts, feature_stale = CANDLE_CACHE.latest_feature_row(asset_u, timeframe)
    spot = float(feature_spot)
    as_of_ts = feature_as_of_ts
    stale = bool(feature_stale)
    try:
        live_spot, live_as_of_ts, live_stale = CANDLE_CACHE.latest_spot_quote(asset_u)
        spot = float(live_spot)
        as_of_ts = live_as_of_ts
        stale = bool(feature_stale or live_stale)
    except Exception:
        # If live quote is unavailable, keep timeframe-derived spot.
        pass

    response = _build_prediction_from_snapshot(
        asset=asset_u,
        horizon=horizon_u,
        model_version=resolved_model_version,
        bundle=bundle,
        feature_row=feature_row,
        price_spot=spot,
        as_of=as_of_ts,
        stale_data=stale,
    )
    return PredictionContext(
        response=response,
        bundle=bundle,
        model_version=resolved_model_version,
    )


def _predict_internal(asset: str, horizon: str) -> PredictResponse:
    return _predict_internal_ctx(asset=asset, horizon=horizon).response


def _quantile_percent_from_key(key: str) -> int | None:
    raw = str(key or "").strip().lower()
    if not raw.startswith("q"):
        return None
    digits = raw[1:]
    if not digits.isdigit():
        return None
    percent = int(digits)
    if percent <= 0 or percent >= 100:
        return None
    return percent


def _interval_score_value(*, lower: float, upper: float, actual: float, coverage: float) -> float:
    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        lo, hi = hi, lo
    coverage_value = max(0.0, min(0.999999, float(coverage)))
    alpha = max(1e-12, 1.0 - coverage_value)
    width = max(0.0, hi - lo)
    if actual < lo:
        return width + ((2.0 / alpha) * (lo - actual))
    if actual > hi:
        return width + ((2.0 / alpha) * (actual - hi))
    return width


def _nominal_coverage_from_prediction(prediction: dict[str, Any]) -> float:
    price_range = prediction.get("price_range", {})
    if isinstance(price_range, dict):
        try:
            return max(0.0, min(0.999999, float(price_range.get("nominal", 0.80))))
        except (TypeError, ValueError):
            return 0.80
    return 0.80


def _weighted_interval_score(event: dict[str, Any], actual_price: float) -> float | None:
    prediction = event.get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    pred_mid = float(event["pred_mid"])

    intervals: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()

    def append_interval(*, coverage: float, lower: float, upper: float) -> None:
        coverage_value = max(0.0, min(0.999999, float(coverage)))
        lo = float(lower)
        hi = float(upper)
        if hi < lo:
            lo, hi = hi, lo
        key = (round(coverage_value, 6), round(lo, 8), round(hi, 8))
        if key in seen:
            return
        seen.add(key)
        intervals.append((coverage_value, lo, hi))

    quantiles_pred = prediction.get("quantiles_pred", {})
    quantile_prices: dict[int, float] = {}
    if isinstance(quantiles_pred, dict):
        for key, value in quantiles_pred.items():
            percent = _quantile_percent_from_key(str(key))
            if percent is None:
                continue
            try:
                quantile_prices[percent] = float(value)
            except (TypeError, ValueError):
                continue

    for low_percent in sorted(percent for percent in quantile_prices if percent < 50):
        high_percent = 100 - low_percent
        if high_percent not in quantile_prices:
            continue
        append_interval(
            coverage=(high_percent - low_percent) / 100.0,
            lower=quantile_prices[low_percent],
            upper=quantile_prices[high_percent],
        )

    append_interval(
        coverage=_nominal_coverage_from_prediction(prediction),
        lower=float(event["pred_low"]),
        upper=float(event["pred_high"]),
    )

    if not intervals:
        return None

    median = quantile_prices.get(50, pred_mid)
    weighted_sum = 0.5 * abs(actual_price - median)
    for coverage, lower, upper in intervals:
        alpha = max(1e-12, 1.0 - coverage)
        weighted_sum += (alpha / 2.0) * _interval_score_value(
            lower=lower,
            upper=upper,
            actual=actual_price,
            coverage=coverage,
        )
    return weighted_sum / (len(intervals) + 0.5)


def _build_event_metrics(event: dict[str, Any], actual_price: float, now_utc: datetime) -> dict[str, Any]:
    pred_low = float(event["pred_low"])
    pred_mid = float(event["pred_mid"])
    pred_high = float(event["pred_high"])
    price_t0 = float(event["price_t0"])
    prediction = event.get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    hit = pred_low <= actual_price <= pred_high
    if actual_price < pred_low:
        miss_side = "below"
    elif actual_price > pred_high:
        miss_side = "above"
    else:
        miss_side = "inside"
    abs_error = abs(actual_price - pred_mid)
    rel_error = abs_error / max(price_t0, 1e-12)
    width_pct = ((pred_high - pred_low) / max(price_t0, 1e-12)) * 100.0
    interval_score = _interval_score_value(
        lower=pred_low,
        upper=pred_high,
        actual=actual_price,
        coverage=_nominal_coverage_from_prediction(prediction),
    )
    wis = _weighted_interval_score(event, actual_price)
    latency_ms = max(0, int((now_utc - EventStore.parse_utc(event["expires_at"])).total_seconds() * 1000))
    return {
        "hit": hit,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "width_pct": width_pct,
        "interval_score": interval_score,
        "wis": wis,
        "miss_side": miss_side,
        "center_error": actual_price - pred_mid,
        "latency_ms": latency_ms,
    }


def _spawn_tournament_run(cycle: dict[str, Any]) -> dict[str, Any] | None:
    model_ids = [str(item) for item in cycle.get("model_ids", []) if str(item)]
    if not model_ids:
        raise ValueError(f"tournament cycle has no model_ids: {cycle['cycle_id']}")
    created_at_run = datetime.now(timezone.utc)
    total_runs = int(cycle.get("total_runs", 0))
    seq = int(cycle.get("launched_runs", 0)) + 1
    cycle_id = str(cycle["cycle_id"])
    asset = str(cycle["asset"]).upper()
    horizon = str(cycle["horizon"])

    bundles_by_id: dict[str, ModelBundle] = {}
    timeframes_by_id: dict[str, str] = {}
    feature_snapshots: dict[str, tuple[dict[str, Any], float, Any, bool]] = {}
    fallback_spot: float | None = None
    fallback_as_of: Any = created_at_run
    stale_flags: list[bool] = []

    for model_id in model_ids:
        bundle = _load_bundle_by_id_cached(asset, horizon, model_id)
        if bundle is None:
            raise ValueError(f"model_id not found for tournament {asset}/{horizon}: {model_id}")
        bundles_by_id[model_id] = bundle
        timeframe = _required_timeframe_for_model(
            asset=asset,
            horizon=horizon,
            model_id=model_id,
            bundle=bundle,
        )
        timeframes_by_id[model_id] = timeframe
        if timeframe not in feature_snapshots:
            feature_row, feature_spot, feature_as_of, feature_stale = CANDLE_CACHE.latest_feature_row(asset, timeframe)
            feature_snapshots[timeframe] = (feature_row, float(feature_spot), feature_as_of, bool(feature_stale))
            stale_flags.append(bool(feature_stale))
            if fallback_spot is None:
                fallback_spot = float(feature_spot)
                fallback_as_of = feature_as_of

    shared_spot = float(fallback_spot if fallback_spot is not None else 0.0)
    shared_as_of = fallback_as_of
    try:
        live_spot, live_as_of, live_stale = CANDLE_CACHE.latest_spot_quote(asset)
        shared_spot = float(live_spot)
        shared_as_of = live_as_of
        stale_flags.append(bool(live_stale))
    except Exception:
        if fallback_spot is None:
            raise
    shared_stale = any(stale_flags)

    batch_events: list[EventCreateInput] = []
    for model_id in model_ids:
        bundle = bundles_by_id[model_id]
        timeframe = timeframes_by_id[model_id]
        feature_row = feature_snapshots[timeframe][0]
        response = _build_prediction_from_snapshot(
            asset=asset,
            horizon=horizon,
            model_version=model_id,
            bundle=bundle,
            feature_row=feature_row,
            price_spot=shared_spot,
            as_of=shared_as_of,
            stale_data=shared_stale,
        )
        batch_events.append(
            EventCreateInput(
                asset=response.asset,
                horizon=response.horizon,
                created_at=created_at_run,
                price_t0=float(response.price_spot),
                prediction=response.model_dump(),
                pred_low=float(response.price_range.low),
                pred_mid=float(response.median_price),
                pred_high=float(response.price_range.high),
                model_id=model_id,
                model_meta=_model_meta_from_bundle(bundle, model_id),
                price_source=str(cycle.get("price_source", "binance_spot")),
                note=cycle.get("note"),
                cycle_id=cycle_id,
                cycle_seq=seq,
                cycle_total=total_runs,
                params_snapshot=_event_params_snapshot(
                    bundle=bundle,
                    model_id=model_id,
                    asset=response.asset,
                    horizon=response.horizon,
                    price_source=str(cycle.get("price_source", "binance_spot")),
                ),
            )
        )

    created = EVENT_STORE.create_events_batch_for_cycle(
        cycle_id,
        seq,
        total_runs,
        batch_events,
    )
    return created[0] if created else None


def _spawn_cycle_event(cycle: dict[str, Any]) -> dict[str, Any] | None:
    total_runs = int(cycle.get("total_runs", 0))
    launched_runs = int(cycle.get("launched_runs", 0))
    if launched_runs >= total_runs:
        return None
    if EVENT_STORE.has_active_event_for_cycle(str(cycle["cycle_id"])):
        return None
    if str(cycle.get("mode", "single")) == "tournament":
        return _spawn_tournament_run(cycle)

    requested_model_id = cycle.get("model_id") or None
    ctx = _predict_internal_ctx(
        asset=str(cycle["asset"]),
        horizon=str(cycle["horizon"]),
        model_id=str(requested_model_id) if requested_model_id else None,
    )
    model_meta = _model_meta_from_bundle(ctx.bundle, ctx.model_version)
    params_snapshot = _event_params_snapshot(
        bundle=ctx.bundle,
        model_id=ctx.model_version,
        asset=ctx.response.asset,
        horizon=ctx.response.horizon,
        price_source=str(cycle.get("price_source", "binance_spot")),
    )
    seq = launched_runs + 1
    return EVENT_STORE.create_event(
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
            price_source=str(cycle.get("price_source", "binance_spot")),
            note=cycle.get("note"),
            cycle_id=str(cycle["cycle_id"]),
            cycle_seq=seq,
            cycle_total=total_runs,
            params_snapshot=params_snapshot,
        )
    )


def _try_spawn_cycle_events(limit: int = 200) -> None:
    for cycle in EVENT_STORE.list_running_cycles(limit=limit):
        cycle_id = str(cycle["cycle_id"])
        try:
            if EVENT_STORE.has_active_event_for_cycle(cycle_id):
                continue
            _spawn_cycle_event(cycle)
        except Exception as exc:
            EVENT_STORE.add_alert(
                level="error",
                code="cycle_spawn_failed",
                message=str(exc),
                context={"cycle_id": cycle_id},
            )


def _run_tournament_orchestrator(limit: int = 200) -> None:
    rules = _tournament_orchestrator_rules()
    enabled = bool(rules.get("enabled", True))
    if not enabled:
        return
    for cycle in EVENT_STORE.list_running_cycles(limit=limit):
        if str(cycle.get("mode", "single")) != "tournament":
            continue
        cycle_id = str(cycle["cycle_id"])
        try:
            summary = EVENT_STORE.cycle_summary(cycle_id, orchestrator_rules=rules)
            if not bool(summary.get("early_stop_triggered", False)):
                continue
            recommendation = str(summary.get("recommendation", "continue")).strip().lower()
            if recommendation not in {"stop", "winner"}:
                continue
            changed = EVENT_STORE.apply_tournament_early_stop(cycle_id)
            if not changed:
                continue
            EVENT_STORE.add_alert(
                level="info",
                code="tournament_orchestrator_stop",
                message=f"early stop applied for cycle {cycle_id}",
                context={
                    "cycle_id": cycle_id,
                    "recommendation": recommendation,
                    "winner_model_id": summary.get("winner_model_id"),
                    "stop_reason": summary.get("stop_reason"),
                    "paired_runs": summary.get("paired_runs"),
                },
            )
        except KeyError:
            continue
        except Exception as exc:
            EVENT_STORE.add_alert(
                level="error",
                code="tournament_orchestrator_failed",
                message=str(exc),
                context={"cycle_id": cycle_id},
            )


def _worker_tick() -> None:
    active = EVENT_STORE.list_active_events()
    now_utc = datetime.now(timezone.utc)
    price_cache: dict[tuple[str, str], tuple[float, Any, bool]] = {}
    for event in active:
        try:
            params_json = event.get("params_json", {})
            base_timeframe = "1m"
            if isinstance(params_json, dict):
                candidate = str(params_json.get("base_timeframe", "")).strip()
                if candidate:
                    base_timeframe = candidate
            cache_key = (str(event["asset"]).upper(), base_timeframe)
            if cache_key not in price_cache:
                price_cache[cache_key] = CANDLE_CACHE.latest_price(cache_key[0], timeframe=base_timeframe)
            current_price, sample_ts, stale = price_cache[cache_key]
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
    _run_tournament_orchestrator()
    _try_spawn_cycle_events()


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


def _job_worker_tick() -> bool:
    job = EVENT_STORE.claim_next_job(job_types=["train_model", "sweep_train", "wf_eval"])
    if not job:
        return False

    job_id = str(job["job_id"])
    job_type = str(job.get("type", "train_model"))

    try:
        if job_type == "wf_eval":
            result = run_walk_forward_job(
                job=job,
                store=EVENT_STORE,
                base_config=CONFIG,
                root=CONFIG_ROOT,
            )
        else:
            result = run_training_job(
                job=job,
                store=EVENT_STORE,
                base_config=CONFIG,
                root=CONFIG_ROOT,
                sync_models=_sync_models_registry,
            )
        _MODEL_BUNDLES.clear()
        _MODEL_BUNDLES_BY_ID.clear()
        EVENT_STORE.add_job_log(job_id, message="job succeeded", stage="done")
        EVENT_STORE.update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            stage="done",
            result=result,
            error_text=None,
            finished=True,
        )
    except TrainingJobCancelled as exc:
        EVENT_STORE.add_job_log(job_id, message=str(exc), level="warning", stage="done")
        EVENT_STORE.update_job(
            job_id,
            status="canceled",
            progress=1.0,
            stage="done",
            error_text=str(exc),
            finished=True,
        )
    except Exception as exc:
        EVENT_STORE.add_job_log(job_id, message=str(exc), level="error", stage="done")
        EVENT_STORE.add_alert(
            level="error",
            code="training_job_failed",
            message=str(exc),
            context={"job_id": job_id},
        )
        EVENT_STORE.update_job(
            job_id,
            status="failed",
            progress=1.0,
            stage="done",
            error_text=str(exc),
            finished=True,
        )
    return True


def _job_worker_loop() -> None:
    while not _JOB_WORKER_STOP.is_set():
        processed = _job_worker_tick()
        if processed:
            continue
        _JOB_WORKER_STOP.wait(1)


def _start_job_worker() -> None:
    global _JOB_WORKER_THREAD
    with _JOB_WORKER_LOCK:
        if _JOB_WORKER_THREAD and _JOB_WORKER_THREAD.is_alive():
            return
        _JOB_WORKER_STOP.clear()
        _JOB_WORKER_THREAD = Thread(target=_job_worker_loop, daemon=True, name="training-job-worker")
        _JOB_WORKER_THREAD.start()


def _stop_job_worker() -> None:
    with _JOB_WORKER_LOCK:
        _JOB_WORKER_STOP.set()
        if _JOB_WORKER_THREAD and _JOB_WORKER_THREAD.is_alive():
            _JOB_WORKER_THREAD.join(timeout=5)


def get_health() -> HealthResponse:
    STATS.total_requests += 1
    missing_models: list[str] = []
    for asset in SUPPORTED_ASSETS:
        for horizon in SUPPORTED_HORIZONS:
            if _load_bundle_cached(asset, horizon) is None:
                missing_models.append(f"{asset}_{horizon}")
    return HealthResponse(
        ok=True,
        status="ok",
        mode="price_ranges",
        fallback_enabled=bool(CONFIG.get("data", {}).get("allow_offline_synthetic_fallback", True)),
        fallback_in_use=bool(missing_models),
        missing_models=missing_models,
        ts_utc=datetime.now(timezone.utc).isoformat(),
    )


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


def _create_builder_jobs(request: TrainingJobCreateRequest, *, job_type: str) -> list[dict[str, Any]]:
    validation = validate_builder_payload(request.model_dump(), require_confirmation=True, job_type=job_type)
    if not validation.response["ok"]:
        raise ValueError("; ".join(validation.response["errors"]))

    payload = request.model_dump()
    payload["asset"] = validation.normalized["asset"]
    payload["horizons"] = list(validation.normalized["horizons"])
    summary = (
        f"{job_type} {validation.normalized['asset']} "
        f"{', '.join(validation.normalized['horizons'])} "
        f"{validation.normalized['base_timeframe_mode']} "
        f"{'/'.join(validation.response['quantiles_final'])}"
    )
    if job_type == "sweep_train":
        variant_count = estimate_sweep_variant_count(payload)
        summary = f"{summary} [{variant_count} variants]"
    payload["job_summary_hint"] = summary
    return [EVENT_STORE.create_job(job_type=job_type, params=payload)]


def _artifact_dir_for_model(model: dict[str, Any]) -> Path:
    return PATHS.artifacts_root / "models" / str(model["asset"]).upper() / str(model["horizon"]) / str(model["model_id"])


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    _sync_models_registry()
    _try_spawn_cycle_events()
    _start_worker()
    enable_inprocess_job_worker = _env_flag("ULYANAI_ENABLE_INPROCESS_JOB_WORKER", False)
    if enable_inprocess_job_worker:
        _start_job_worker()
    try:
        yield
    finally:
        if enable_inprocess_job_worker:
            _stop_job_worker()
        _stop_worker()


app = FastAPI(title="UlyanAI Inference Service", version="0.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_ui() -> FileResponse:
    ui_path = PATHS.root / "web" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(ui_path, media_type="text/html; charset=utf-8")


@app.get("/styles.css", include_in_schema=False)
def ui_styles() -> FileResponse:
    css_path = PATHS.root / "web" / "styles.css"
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="web/styles.css not found")
    return FileResponse(css_path, media_type="text/css; charset=utf-8")


@app.get("/app.js", include_in_schema=False)
def ui_script() -> FileResponse:
    js_path = PATHS.root / "web" / "app.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="web/app.js not found")
    return FileResponse(js_path, media_type="application/javascript; charset=utf-8")


@app.get("/music.mp3", include_in_schema=False)
def ui_music() -> FileResponse:
    cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    preferred_paths = (
        PATHS.root / "web" / "music.mp3",
        PATHS.root / "music.mp3",
        PATHS.root / "FRIENDLY_THUG_52_NGG_-_No_Gletcher_Gang_76929086.mp3",
    )
    for audio_path in preferred_paths:
        if audio_path.exists():
            return FileResponse(audio_path, media_type="audio/mpeg", headers=cache_headers)
    for audio_path in sorted(PATHS.root.glob("*.mp3")):
        if audio_path.is_file():
            return FileResponse(audio_path, media_type="audio/mpeg", headers=cache_headers)
    raise HTTPException(status_code=404, detail="no mp3 file found in project root")


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
        params_snapshot = _event_params_snapshot(
            bundle=ctx.bundle,
            model_id=ctx.model_version,
            asset=ctx.response.asset,
            horizon=ctx.response.horizon,
            price_source=request.price_source,
        )
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
                params_snapshot=params_snapshot,
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


@app.post("/api/cycles", response_model=CycleRecord)
def create_cycle(
    request: CreateCycleRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CycleRecord:
    try:
        asset_u, horizon_u = _check_supported(request.asset, request.horizon)
        requested_model_ids = [str(item).strip() for item in list(request.model_ids or []) if str(item).strip()]
        mode = str(request.mode or "").strip().lower() if request.mode else ""
        if not mode:
            mode = "tournament" if len(requested_model_ids) >= 2 else "single"
        if mode not in {"single", "tournament"}:
            raise ValueError(f"unsupported cycle mode: {request.mode}")
        model_id = request.model_id
        if mode == "tournament":
            if len(requested_model_ids) < 2:
                raise ValueError("tournament mode requires at least 2 model_ids")
            if len(requested_model_ids) > _max_models_per_tournament():
                raise ValueError(f"tournament model_ids exceeds limit {_max_models_per_tournament()}")
            validated_model_ids: list[str] = []
            seen_model_ids: set[str] = set()
            for candidate in requested_model_ids:
                if candidate in seen_model_ids:
                    continue
                seen_model_ids.add(candidate)
                model = EVENT_STORE.get_model(candidate)
                if str(model.get("asset", "")).upper() != asset_u or str(model.get("horizon", "")) != horizon_u:
                    raise ValueError(f"model_id does not match cycle asset/horizon: {candidate}")
                if _load_bundle_by_id_cached(asset_u, horizon_u, candidate) is None:
                    raise ValueError(f"model artifact not found for cycle: {candidate}")
                validated_model_ids.append(candidate)
            if len(validated_model_ids) < 2:
                raise ValueError("tournament mode requires at least 2 distinct model_ids")
            requested_model_ids = validated_model_ids
            model_id = None
        else:
            if not model_id and len(requested_model_ids) == 1:
                model_id = requested_model_ids[0]
            if model_id:
                model = EVENT_STORE.get_model(model_id)
                if str(model.get("asset", "")).upper() != asset_u or str(model.get("horizon", "")) != horizon_u:
                    raise ValueError(f"model_id does not match cycle asset/horizon: {model_id}")
                if _load_bundle_by_id_cached(asset_u, horizon_u, model_id) is None:
                    raise ValueError(f"model artifact not found for cycle: {model_id}")
            requested_model_ids = []
        cycle = EVENT_STORE.create_cycle(
            CycleCreateInput(
                asset=asset_u,
                horizon=horizon_u,
                total_runs=int(request.runs),
                model_id=model_id,
                mode=mode,
                model_ids=tuple(requested_model_ids),
                models_count=len(requested_model_ids) if requested_model_ids else 1,
                align_to_boundary=bool(request.align_to_boundary),
                price_source=request.price_source,
                note=request.note,
            )
        )
        _spawn_cycle_event(cycle)
        return CycleRecord.model_validate(EVENT_STORE.get_cycle(str(cycle["cycle_id"])))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        EVENT_STORE.add_alert(
            level="error",
            code="create_cycle_failed",
            message=str(exc),
            context={"asset": request.asset, "horizon": request.horizon, "runs": request.runs},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/cycles", response_model=CycleListResponse)
def list_cycles(
    status: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CycleListResponse:
    payload = EVENT_STORE.list_cycles(status=status, horizon=horizon, page=page, page_size=page_size)
    return CycleListResponse.model_validate(payload)


@app.get("/api/cycles/{cycle_id}", response_model=CycleRecord)
def get_cycle(
    cycle_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CycleRecord:
    try:
        return CycleRecord.model_validate(EVENT_STORE.get_cycle(cycle_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/cycles/{cycle_id}/summary", response_model=TournamentCycleSummaryResponse)
def get_cycle_summary(
    cycle_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TournamentCycleSummaryResponse:
    try:
        payload = EVENT_STORE.cycle_summary(cycle_id, orchestrator_rules=_tournament_orchestrator_rules())
        return TournamentCycleSummaryResponse.model_validate(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/events", response_model=EventListResponse)
def list_events(
    status: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    result: str | None = Query(default=None, pattern="^(hit|miss)?$"),
    model_id: str | None = Query(default=None),
    model_ids: str | None = Query(default=None),
    exclude_stale: bool = Query(default=False),
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
    model_id_list = [item.strip() for item in str(model_ids or "").split(",") if item.strip()]
    payload = EVENT_STORE.list_events(
        status=status,
        horizon=horizon,
        cycle_id=cycle_id,
        model_id=model_id,
        model_ids=model_id_list or None,
        exclude_stale=exclude_stale,
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
    include_archived: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelListResponse:
    _sync_models_registry()
    statuses = ["active"]
    if include_archived:
        statuses.append("archived")
    if include_deleted:
        statuses.append("deleted")
    return ModelListResponse.model_validate(
        {
            "items": EVENT_STORE.list_models(
                asset=asset,
                horizon=horizon,
                statuses=statuses,
                limit=limit,
            )
        }
    )


@app.get("/api/models/by-id/{model_id}", response_model=ModelRecord)
def get_model(
    model_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        _sync_models_registry()
        return ModelRecord.model_validate(EVENT_STORE.get_model(model_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/models/production", response_model=ProductionModelsResponse)
def production_models(
    asset: str = Query(default="BTC"),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ProductionModelsResponse:
    _sync_models_registry()
    return ProductionModelsResponse.model_validate({"items": EVENT_STORE.production_models(asset=asset)})


@app.post("/api/models/{model_id}/metadata", response_model=ModelRecord)
def update_model_metadata(
    model_id: str,
    request: ModelMetadataUpdateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        updated = EVENT_STORE.update_model_management(
            model_id,
            notes=request.notes,
            tags=request.tags,
            parent_model_id=request.parent_model_id,
            experiment_id=request.experiment_id,
        )
        return ModelRecord.model_validate(updated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/promote", response_model=ModelRecord)
def promote_model(
    model_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        model = EVENT_STORE.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if model["status"] != "active":
        raise HTTPException(status_code=400, detail="only active models can be promoted")
    EVENT_STORE.set_production_model(str(model["asset"]), str(model["horizon"]), model_id)
    _MODEL_BUNDLES.pop((str(model["asset"]).upper(), str(model["horizon"])), None)
    return ModelRecord.model_validate(EVENT_STORE.get_model(model_id))


@app.post("/api/models/{model_id}/archive", response_model=ModelRecord)
def archive_model(
    model_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        updated = EVENT_STORE.update_model_management(model_id, status="archived")
        return ModelRecord.model_validate(updated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/unarchive", response_model=ModelRecord)
def unarchive_model(
    model_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        updated = EVENT_STORE.update_model_management(model_id, status="active")
        return ModelRecord.model_validate(updated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/delete", response_model=ModelRecord)
def soft_delete_model(
    model_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    try:
        updated = EVENT_STORE.update_model_management(model_id, status="deleted")
        return ModelRecord.model_validate(updated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/purge", response_model=ModelRecord)
def purge_model(
    model_id: str,
    confirm: bool = Query(default=False),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelRecord:
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required for purge")
    try:
        model = EVENT_STORE.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if int(model.get("event_refs", 0)) > 0:
        raise HTTPException(status_code=400, detail="cannot purge model referenced by events")
    artifact_dir = _artifact_dir_for_model(model)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    EVENT_STORE.purge_model_row(model_id)
    return ModelRecord.model_validate(model)


@app.post("/api/lab/validate", response_model=ModelBuilderValidateResponse)
def validate_model_builder(
    request: ModelBuilderValidateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> ModelBuilderValidateResponse:
    validation = validate_builder_payload(request.model_dump(), require_confirmation=False)
    return ModelBuilderValidateResponse.model_validate(validation.response)


@app.post("/api/jobs/train_model", response_model=JobListResponse)
def create_train_model_job(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobListResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="train_model")
        return JobListResponse.model_validate({"items": jobs})
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/sweep_train", response_model=JobListResponse)
def create_sweep_train_job(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobListResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="sweep_train")
        return JobListResponse.model_validate({"items": jobs})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/wf_eval", response_model=JobListResponse)
def create_wf_eval_job(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobListResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="wf_eval")
        return JobListResponse.model_validate({"items": jobs})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs", response_model=JobListResponse)
def list_jobs(
    statuses: str | None = Query(default=None),
    types: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobListResponse:
    status_list = [item.strip() for item in str(statuses or "").split(",") if item.strip()]
    type_list = [item.strip() for item in str(types or "").split(",") if item.strip()]
    return JobListResponse.model_validate(
        {
            "items": EVENT_STORE.list_jobs(
                statuses=status_list or None,
                job_types=type_list or None,
                limit=limit,
            )
        }
    )


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(
    job_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobRecord:
    try:
        return JobRecord.model_validate(EVENT_STORE.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/logs", response_model=JobLogsResponse)
def get_job_logs(
    job_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobLogsResponse:
    try:
        job = EVENT_STORE.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobLogsResponse.model_validate(
        {
            "job": job,
            "logs": EVENT_STORE.list_job_logs(job_id, limit=limit),
        }
    )


@app.post("/api/jobs/{job_id}/cancel", response_model=JobRecord)
def cancel_job(
    job_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> JobRecord:
    try:
        return JobRecord.model_validate(EVENT_STORE.request_cancel_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/lab/jobs/train", response_model=TrainingJobBatchResponse)
def create_train_job(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobBatchResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="train_model")
        return TrainingJobBatchResponse.model_validate(
            {"items": [EVENT_STORE.get_training_job(str(item["job_id"])) for item in jobs]}
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/lab/jobs/sweep", response_model=TrainingJobBatchResponse)
def create_sweep_jobs(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobBatchResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="sweep_train")
        return TrainingJobBatchResponse.model_validate(
            {"items": [EVENT_STORE.get_training_job(str(item["job_id"])) for item in jobs]}
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/lab/jobs/wf_eval", response_model=TrainingJobBatchResponse)
def create_wf_eval_jobs(
    request: TrainingJobCreateRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobBatchResponse:
    try:
        jobs = _create_builder_jobs(request, job_type="wf_eval")
        return TrainingJobBatchResponse.model_validate(
            {"items": [EVENT_STORE.get_training_job(str(item["job_id"])) for item in jobs]}
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/lab/jobs", response_model=TrainingJobListResponse)
def list_training_jobs(
    statuses: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobListResponse:
    status_list = [item.strip() for item in str(statuses or "").split(",") if item.strip()]
    return TrainingJobListResponse.model_validate(
        {
            "items": EVENT_STORE.list_training_jobs(
                statuses=status_list or None,
                limit=limit,
            )
        }
    )


@app.get("/api/lab/jobs/{job_id}", response_model=TrainingJobRecord)
def get_training_job(
    job_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobRecord:
    try:
        return TrainingJobRecord.model_validate(EVENT_STORE.get_training_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/lab/jobs/{job_id}/logs", response_model=TrainingJobLogsResponse)
def get_training_job_logs(
    job_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobLogsResponse:
    try:
        job = EVENT_STORE.get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TrainingJobLogsResponse.model_validate(
        {
            "job": job,
            "logs": EVENT_STORE.list_training_job_logs(job_id, limit=limit),
        }
    )


@app.post("/api/lab/jobs/{job_id}/cancel", response_model=TrainingJobRecord)
def cancel_training_job(
    job_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobRecord:
    try:
        return TrainingJobRecord.model_validate(EVENT_STORE.request_cancel_training_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/lab/jobs/{job_id}/retry", response_model=TrainingJobRecord)
def retry_training_job(
    job_id: str,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> TrainingJobRecord:
    try:
        existing = EVENT_STORE.get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    retry_job = EVENT_STORE.create_training_job(
        job_kind=str(existing.get("job_kind", "train")),
        asset=str(existing.get("asset", "BTC")),
        horizons=[str(item) for item in existing.get("horizons", [])],
        summary=f"retry of {job_id}",
        params=dict(existing.get("params", {})),
    )
    return TrainingJobRecord.model_validate(retry_job)


@app.post("/api/reports/generate", response_model=GenerateReportResponse)
def generate_report(
    request: GenerateReportRequest,
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> GenerateReportResponse:
    try:
        payload = generate_experiment_report_artifacts(
            store=EVENT_STORE,
            artifacts_root=PATHS.artifacts_root,
            experiment_id=request.experiment_id,
            cycle_id=request.cycle_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    experiment_quoted = quote(str(payload["experiment_id"]), safe="")
    payload["report_pdf_url"] = f"/api/reports/{experiment_quoted}/download?format=pdf"
    payload["report_txt_url"] = f"/api/reports/{experiment_quoted}/download?format=txt"
    return GenerateReportResponse.model_validate(payload)


@app.get("/api/reports/{experiment_id}/download")
def download_report(
    experiment_id: str,
    format: str = Query(default="pdf", pattern="^(pdf|txt)$"),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> FileResponse:
    try:
        _, txt_path, pdf_path = resolve_report_paths(PATHS.artifacts_root, experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    format_clean = str(format).strip().lower()
    target_path = pdf_path if format_clean == "pdf" else txt_path
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    media_type = "application/pdf" if format_clean == "pdf" else "text/plain; charset=utf-8"
    download_name = f"{experiment_id}_report.{format_clean}"
    return FileResponse(target_path, media_type=media_type, filename=download_name)


@app.get("/api/markets/polymarket/crypto", response_model=PolymarketCryptoEventsResponse)
def polymarket_crypto_events(
    active: bool | None = Query(default=True),
    closed: bool | None = Query(default=False),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> PolymarketCryptoEventsResponse:
    try:
        raw_items = _fetch_polymarket_events(
            tag_slug=POLYMARKET_CRYPTO_TAG_SLUG,
            active=active,
            closed=closed,
        )
        items = [_normalize_polymarket_event(item) for item in raw_items]
        return PolymarketCryptoEventsResponse.model_validate(
            {
                "source": "polymarket_gamma",
                "tag_slug": POLYMARKET_CRYPTO_TAG_SLUG,
                "active": active,
                "closed": closed,
                "total": len(items),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "items": items,
            }
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"polymarket gamma request failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    days: int = Query(default=30, ge=1, le=3650),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> MetricsSummaryResponse:
    return MetricsSummaryResponse.model_validate(
        EVENT_STORE.metrics_summary(days=days, horizon=horizon, cycle_id=cycle_id)
    )


@app.get("/api/metrics/compare", response_model=CompareMetricsResponse)
def metrics_compare(
    days: int = Query(default=30, ge=1, le=3650),
    asset: str = Query(default="BTC"),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    mode: str = Query(default="simple", pattern="^(simple|matched|grouped)$"),
    model_ids: str | None = Query(default=None),
    exclude_stale: bool = Query(default=False),
    group_limit: int = Query(default=200, ge=1, le=1000),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CompareMetricsResponse:
    selected_model_ids = [item.strip() for item in str(model_ids or "").split(",") if item.strip()]
    return CompareMetricsResponse.model_validate(
        EVENT_STORE.compare_metrics(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=selected_model_ids or None,
            mode=mode,
            exclude_stale=exclude_stale,
            group_limit=group_limit,
        )
    )


@app.get("/api/compare/scoreboard", response_model=CompareScoreboardResponse)
def compare_scoreboard(
    days: int = Query(default=30, ge=1, le=3650),
    asset: str | None = Query(default="BTC"),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    mode: str = Query(default="simple", pattern="^(simple|matched|grouped)$"),
    model_ids: str | None = Query(default=None),
    exclude_stale: bool = Query(default=False),
    baseline_model_id: str | None = Query(default=None),
    columns: str | None = Query(default=None),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CompareScoreboardResponse:
    selected_model_ids = [item.strip() for item in str(model_ids or "").split(",") if item.strip()]
    selected_columns = [item.strip() for item in str(columns or "").split(",") if item.strip()]
    return CompareScoreboardResponse.model_validate(
        EVENT_STORE.compare_scoreboard(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=selected_model_ids or None,
            mode=mode,
            exclude_stale=exclude_stale,
            baseline_model_id=baseline_model_id,
            columns=selected_columns or None,
        )
    )


@app.get("/api/compare/series", response_model=CompareSeriesResponse)
def compare_series(
    days: int = Query(default=30, ge=1, le=3650),
    asset: str | None = Query(default="BTC"),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    mode: str = Query(default="simple", pattern="^(simple|matched|grouped)$"),
    model_ids: str | None = Query(default=None),
    exclude_stale: bool = Query(default=False),
    metric: str = Query(default="coverage"),
    rolling_days: int = Query(default=7, ge=1, le=365),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CompareSeriesResponse:
    selected_model_ids = [item.strip() for item in str(model_ids or "").split(",") if item.strip()]
    return CompareSeriesResponse.model_validate(
        EVENT_STORE.compare_series(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=selected_model_ids or None,
            mode=mode,
            exclude_stale=exclude_stale,
            metric=metric,
            rolling_days=rolling_days,
        )
    )


@app.get("/api/compare/hist", response_model=CompareHistogramResponse)
def compare_hist(
    days: int = Query(default=30, ge=1, le=3650),
    asset: str | None = Query(default="BTC"),
    horizon: str | None = Query(default=None),
    cycle_id: str | None = Query(default=None),
    mode: str = Query(default="simple", pattern="^(simple|matched|grouped)$"),
    model_ids: str | None = Query(default=None),
    exclude_stale: bool = Query(default=False),
    metric: str = Query(default="abs_error"),
    bins: int = Query(default=10, ge=1, le=100),
    _: None = Depends(_auth_dep),
    __: None = Depends(_rate_limit_dep),
) -> CompareHistogramResponse:
    selected_model_ids = [item.strip() for item in str(model_ids or "").split(",") if item.strip()]
    return CompareHistogramResponse.model_validate(
        EVENT_STORE.compare_hist(
            days=days,
            asset=asset,
            horizon=horizon,
            cycle_id=cycle_id,
            model_ids=selected_model_ids or None,
            mode=mode,
            exclude_stale=exclude_stale,
            metric=metric,
            bins=bins,
        )
    )


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

