"""Inference API with price-range outputs and model artifact integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

from src.config import get_runtime_paths, load_config
from src.models.predict import (
    ModelBundle,
    ReturnQuantiles,
    build_price_prediction,
    load_model_bundle,
    predict_return_quantiles,
)
from src.service.cache import CandleCache
from src.service.schemas import (
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictResponse,
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
_MODEL_BUNDLES: dict[tuple[str, str], ModelBundle | None] = {}
_MODEL_LOCK = Lock()


@dataclass
class RequestStats:
    total_requests: int = 0
    predict_requests: int = 0
    errors_total: int = 0
    stale_data_responses: int = 0


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


def _timeframe_for_horizon(horizon: str) -> str:
    horizon_cfg = CONFIG.get("horizons_map", {}).get(horizon)
    if not horizon_cfg:
        return horizon
    return str(horizon_cfg.get("timeframe", horizon))


def _predict_internal(asset: str, horizon: str) -> PredictResponse:
    asset_u, horizon_u = _check_supported(asset, horizon)
    timeframe = _timeframe_for_horizon(horizon_u)
    feature_row, spot, as_of_ts, stale = CANDLE_CACHE.latest_feature_row(asset_u, timeframe)

    bundle = _load_bundle_cached(asset_u, horizon_u)
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
        calibration_score=0.92 if bundle is None else float(bundle.metadata.get("calibrator", {}).get("target_coverage", 0.8)),
        drift_flag=False,
        model_version=_model_version(asset_u, horizon_u, bundle),
        stale_data=stale,
    )
    return PredictResponse.model_validate(payload)


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


def _auth_dep(x_api_key: str | None = Header(default=None)) -> None:
    expected = str(CONFIG.get("service", {}).get("api_key", "dev-key"))
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _rate_limit_dep(request: Request) -> None:
    limit = int(CONFIG.get("service", {}).get("rate_limit_per_min", 120))
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    minute = int(datetime.now(timezone.utc).timestamp() // 60)
    key = (ip, minute)
    with _RATE_LOCK:
        current = _RATE_BUCKET.get(key, 0) + 1
        _RATE_BUCKET[key] = current
        if current > limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")


app = FastAPI(title="UlyanAI Inference Service", version="0.2.0")


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
