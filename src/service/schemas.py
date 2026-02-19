"""Pydantic contracts for the inference service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceLevels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p10: float
    p50: float
    p90: float

    @model_validator(mode="after")
    def validate_order(self) -> "PriceLevels":
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("price_levels must satisfy p10 <= p50 <= p90")
        return self


class PriceRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    nominal: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "PriceRange":
        if self.low > self.high:
            raise ValueError("price_range.low must be <= price_range.high")
        return self


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_score: float
    drift_flag: bool


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    as_of: str
    price_spot: float
    mode: Literal["price_ranges"]
    price_levels: PriceLevels
    price_range: PriceRange
    median_price: float
    confidence: Confidence
    model_version: str
    stale_data: bool = False


class PredictBatchRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[PredictBatchRequestItem]


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["price_ranges"]
    assets: list[str]
    horizons: list[str]
    models: dict[str, str]
    data_freshness_seconds: int


class EventModelMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    git_commit: str = "unknown"
    train_time: str = "unknown"
    dataset_hash: str = "unknown"
    config_hash: str = "unknown"


class EventMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    hit: bool | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    width_pct: float | None = None
    latency_ms: int | None = None


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    asset: str
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    created_at: str
    expires_at: str
    price_t0: float
    pred_low: float
    pred_mid: float
    pred_high: float
    prediction: dict
    model_id: str
    model_meta: EventModelMeta
    price_source: str
    status: Literal["active", "completed", "cancelled"]
    actual_price: float | None = None
    current_price: float | None = None
    metrics: EventMetrics = Field(default_factory=EventMetrics)
    note: str | None = None
    updated_at: str


class CreateEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str = "BTC"
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    model_id: str | None = None
    price_source: Literal["binance_spot", "index"] = "binance_spot"
    note: str | None = None


class EventListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EventRecord]
    total: int
    page: int
    page_size: int


class PriceSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    event_id: str
    ts: str
    price: float


class EventPricesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    samples: list[PriceSample]


class ModelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    asset: str
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    name: str
    created_at: str
    git_commit: str
    train_time: str
    dataset_hash: str
    config_hash: str
    metrics_json: dict = Field(default_factory=dict)
    is_production: bool


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ModelRecord]


class ProductionModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ModelRecord]


class MetricsSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    period_days: int
    target_coverage: float
    actual_coverage: float
    coverage_delta: float
    completed_events: int
    active_events: int
    avg_width_pct: float
    avg_abs_error: float
    avg_rel_error: float
    best_horizon: str | None = None
    series: list[dict] = Field(default_factory=list)
    by_horizon: dict = Field(default_factory=dict)


class AlertRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    ts: str
    level: str
    code: str
    message: str
    context_json: dict = Field(default_factory=dict)


class AlertsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AlertRecord]
