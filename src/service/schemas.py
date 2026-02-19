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
