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
    quantiles_pred: dict[str, float] = Field(default_factory=dict)


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

    ok: bool
    status: Literal["ok"] = "ok"
    mode: Literal["price_ranges"] = "price_ranges"
    fallback_enabled: bool = True
    fallback_in_use: bool = False
    missing_models: list[str] = Field(default_factory=list)
    ts_utc: str


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
    interval_score: float | None = None
    wis: float | None = None
    miss_side: Literal["below", "above", "inside"] | None = None
    center_error: float | None = None
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
    cycle_id: str | None = None
    cycle_seq: int | None = None
    cycle_total: int | None = None
    eval_key: str | None = None
    status: Literal["active", "completed", "cancelled"]
    actual_price: float | None = None
    current_price: float | None = None
    metrics: EventMetrics = Field(default_factory=EventMetrics)
    params_json: dict = Field(default_factory=dict)
    note: str | None = None
    updated_at: str


class CreateEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str = "BTC"
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    model_id: str | None = None
    price_source: Literal["binance_spot", "index"] = "binance_spot"
    note: str | None = None


class CreateCycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str = "BTC"
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    runs: int = Field(ge=1, le=500)
    mode: Literal["single", "tournament"] | None = None
    model_id: str | None = None
    model_ids: list[str] = Field(default_factory=list)
    align_to_boundary: bool = False
    price_source: Literal["binance_spot", "index"] = "binance_spot"
    note: str | None = None


class EventListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EventRecord]
    total: int
    page: int
    page_size: int


class CycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_id: str
    asset: str
    horizon: Literal["5m", "15m", "1h", "4h", "1d", "1w"]
    total_runs: int
    launched_runs: int
    completed_runs: int
    cancelled_runs: int
    mode: Literal["single", "tournament"] = "single"
    model_id: str | None = None
    model_ids: list[str] = Field(default_factory=list)
    models_count: int = 1
    align_to_boundary: bool = False
    price_source: str
    status: Literal["running", "completed", "cancelled"]
    note: str | None = None
    last_event_id: str | None = None
    created_at: str
    updated_at: str


class CycleListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CycleRecord]
    total: int
    page: int
    page_size: int


class TournamentLeaderboardRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    model_id: str
    model_name: str | None = None
    total_events: int = 0
    completed_events: int = 0
    paired_runs: int = 0
    wins: int = 0
    win_rate: float | None = None
    coverage: float | None = None
    avg_abs_error: float | None = None
    avg_width_pct: float | None = None


class TournamentPairedMetricRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_seq: int
    created_at: str | None = None
    eval_key: str | None = None
    models_present: list[str] = Field(default_factory=list)
    pair_ready: bool = False
    run_closed: bool = False
    completed_models: int = 0
    cancelled_models: int = 0
    active_models: int = 0
    winner_model_id: str | None = None
    runner_up_model_id: str | None = None
    winner_abs_error: float | None = None
    runner_up_abs_error: float | None = None
    delta_abs_error: float | None = None
    delta_width_pct: float | None = None


class TournamentCycleSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_id: str
    asset: str
    horizon: str
    mode: str
    status: str
    total_runs: int = 0
    launched_runs: int = 0
    completed_runs: int = 0
    cancelled_runs: int = 0
    observed_runs: int = 0
    paired_runs: int = 0
    active_runs: int = 0
    remaining_runs: int = 0
    compared_model_ids: list[str] = Field(default_factory=list)
    recommendation: Literal["continue", "stop", "winner"] = "continue"
    winner_model_id: str | None = None
    stop_reason: str | None = None
    early_stop_triggered: bool = False
    rules: dict = Field(default_factory=dict)
    early_stop_checks: dict = Field(default_factory=dict)
    leaderboard: list[TournamentLeaderboardRow] = Field(default_factory=list)
    paired_metrics: list[TournamentPairedMetricRow] = Field(default_factory=list)
    generated_at: str


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
    status: Literal["active", "archived", "deleted"] = "active"
    deleted_at: str | None = None
    params_json: dict = Field(default_factory=dict)
    notes: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    parent_model_id: str | None = None
    experiment_id: str | None = None
    event_refs: int = 0


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ModelRecord]


class ProductionModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ModelRecord]


class ModelMetadataUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    parent_model_id: str | None = None
    experiment_id: str | None = None


class ModelBuilderRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str = "BTC"
    training_budget_preset: Literal["custom", "F0", "F1", "F2"] = "custom"
    base_timeframe_mode: Literal["legacy", "base_1m"] = "legacy"
    horizons: list[Literal["5m", "15m", "1h", "4h", "1d", "1w"]] = Field(default_factory=lambda: ["5m"])
    steps_overrides: dict[str, int] = Field(default_factory=dict)
    target_coverage_percent: int = Field(default=80, ge=1, le=99)
    interval_mode: Literal["symmetric", "custom", "multi_pack"] = "symmetric"
    custom_q_low_percent: int | None = Field(default=None, ge=1, le=99)
    custom_q_high_percent: int | None = Field(default=None, ge=1, le=99)
    multi_interval_coverages: list[int] = Field(default_factory=list)
    quantile_strategy: Literal["interval_only", "selected_set", "full_grid"] = "interval_only"
    selected_quantiles: list[str] = Field(default_factory=list)
    quantile_soft_limit: int = Field(default=11, ge=1, le=99)
    confirm_resource_heavy: bool = False
    train_window_mode: Literal["expanding", "rolling"] = "expanding"
    train_window_days: int = Field(default=365, ge=1, le=3650)
    val_days: int = Field(default=30, ge=1, le=3650)
    test_days: int = Field(default=30, ge=1, le=3650)
    walk_forward_enabled: bool = False
    wf_train_days: int = Field(default=180, ge=1, le=3650)
    wf_val_days: int = Field(default=30, ge=1, le=3650)
    wf_step_days: int = Field(default=30, ge=1, le=3650)
    wf_folds: int = Field(default=3, ge=1, le=100)
    feature_set_version: str = "feat_v1"
    feature_groups: dict[str, bool] = Field(default_factory=dict)
    feature_overrides: dict[str, list[int] | int | dict[str, int]] = Field(default_factory=dict)
    hyperparams: dict[str, float | int | None] = Field(default_factory=dict)
    calibration_method: Literal["grid_scale", "conformal_cqr"] = "grid_scale"
    scale_grid: list[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0])
    scale_selection_rule: str = "min_abs_coverage_gap_then_width"
    sweep_target_coverages: list[int] = Field(default_factory=list)
    sweep_train_window_days: list[int] = Field(default_factory=list)
    sweep_feature_set_versions: list[str] = Field(default_factory=list)
    sweep_calibration_methods: list[str] = Field(default_factory=list)
    sweep_hyperparams: dict[str, list[float | int]] = Field(default_factory=dict)
    experiment_id: str | None = None
    parent_model_id: str | None = None
    notes: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class ModelBuilderValidateRequest(ModelBuilderRequestBase):
    model_config = ConfigDict(extra="forbid")


class TrainingJobCreateRequest(ModelBuilderRequestBase):
    model_config = ConfigDict(extra="forbid")


class ModelBuilderValidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    requested_coverage: float
    effective_coverage: float
    implied_coverage: float
    q_low: str
    q_high: str
    quantile_soft_limit: int = 11
    quantiles_final: list[str] = Field(default_factory=list)
    quantiles_primary: list[str] = Field(default_factory=list)
    quantiles_multi_interval: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    sweep_variants_count: int = 1


class TrainingJobLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    job_id: str
    ts: str
    level: str
    stage: str | None = None
    message: str


class TrainingJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    job_kind: Literal["train", "sweep", "wf_eval"]
    asset: str
    horizons: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "succeeded", "failed", "canceled"]
    progress: int = 0
    stage: str
    summary: str
    params: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    error: str | None = None
    cancel_requested: bool = False
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class TrainingJobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TrainingJobRecord] = Field(default_factory=list)


class TrainingJobBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TrainingJobRecord] = Field(default_factory=list)


class TrainingJobLogsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: TrainingJobRecord
    logs: list[TrainingJobLogRecord] = Field(default_factory=list)


class GenerateReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    cycle_id: str | None = None


class GenerateReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    cycle_id: str | None = None
    generated_at: str
    models_count: int = 0
    events_count: int = 0
    report_dir: str
    report_pdf_path: str
    report_txt_path: str
    report_pdf_url: str
    report_txt_url: str


class JobLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    job_id: str
    ts: str
    level: str
    stage: str | None = None
    message: str


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    type: Literal["train_model", "sweep_train", "wf_eval"]
    asset: str
    horizons: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "succeeded", "failed", "canceled"]
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: Literal["queued", "data", "features", "train", "calibrate", "register", "done"]
    summary: str
    params: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    error_text: str | None = None
    cancel_requested: bool = False
    logs_text: str = ""
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class JobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[JobRecord] = Field(default_factory=list)


class JobLogsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: JobRecord
    logs: list[JobLogRecord] = Field(default_factory=list)


class PolymarketCryptoMarket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str | None = None
    condition_id: str | None = None
    slug: str | None = None
    question: str | None = None
    active: bool | None = None
    closed: bool | None = None
    end_date: str | None = None
    volume: float | None = None
    volume_clob: float | None = None
    best_ask: float | None = None
    last_trade_price: float | None = None
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    yes_token_id: str | None = None
    no_token_id: str | None = None


class PolymarketCryptoEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    slug: str | None = None
    title: str | None = None
    description: str | None = None
    active: bool | None = None
    closed: bool | None = None
    start_date: str | None = None
    end_date: str | None = None
    volume: float | None = None
    volume_24hr: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    icon: str | None = None
    markets_count: int = 0
    markets: list[PolymarketCryptoMarket] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PolymarketCryptoEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["polymarket_gamma"]
    tag_slug: str
    active: bool | None = None
    closed: bool | None = None
    total: int
    fetched_at: str
    items: list[PolymarketCryptoEvent] = Field(default_factory=list)


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


class CompareModelStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_name: str | None = None
    asset: str | None = None
    horizon: str | None = None
    base_timeframe: str | None = None
    steps_ahead: int | None = None
    quantiles_count: int | None = None
    target_coverage_effective: float | None = None
    status: str | None = None
    is_production: bool = False
    total_events: int = 0
    completed_events: int = 0
    matched_groups: int = 0
    group_wins: int = 0
    coverage: float | None = None
    avg_width_pct: float | None = None
    avg_abs_error: float | None = None
    avg_rel_error: float | None = None
    avg_interval_score: float | None = None
    avg_wis: float | None = None


class CompareGroupRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    created_at: str | None = None
    cycle_id: str | None = None
    event_count: int = 0
    model_ids: list[str] = Field(default_factory=list)
    best_model_id: str | None = None
    best_abs_error: float | None = None
    hit_models: list[str] = Field(default_factory=list)


class CompareMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["simple", "matched", "grouped"] = "simple"
    asset: str = "BTC"
    horizon: str | None = None
    cycle_id: str | None = None
    period_days: int
    total_events: int = 0
    total_completed: int = 0
    total_groups: int = 0
    compared_model_ids: list[str] = Field(default_factory=list)
    items: list[CompareModelStats] = Field(default_factory=list)
    groups: list[CompareGroupRecord] = Field(default_factory=list)


class CompareScoreboardRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_name: str | None = None
    asset: str | None = None
    horizon: str | None = None
    base_timeframe: str | None = None
    steps_ahead: int | None = None
    quantiles_count: int | None = None
    target_coverage_effective: float | None = None
    group_key: str | None = None
    status: str | None = None
    is_production: bool = False
    total_events: int = 0
    completed_events: int = 0
    matched_groups: int = 0
    group_wins: int = 0
    coverage: float | None = None
    avg_width_pct: float | None = None
    avg_abs_error: float | None = None
    avg_rel_error: float | None = None
    avg_interval_score: float | None = None
    avg_wis: float | None = None
    delta_coverage: float | None = None
    delta_avg_width_pct: float | None = None
    delta_avg_abs_error: float | None = None
    delta_avg_rel_error: float | None = None
    delta_avg_interval_score: float | None = None
    delta_avg_wis: float | None = None


class CompareScoreboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["simple", "matched", "grouped"] = "simple"
    asset: str | None = None
    horizon: str | None = None
    cycle_id: str | None = None
    period_days: int
    exclude_stale: bool = False
    baseline_model_id: str | None = None
    compared_model_ids: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    total_events: int = 0
    total_completed: int = 0
    total_groups: int = 0
    items: list[CompareScoreboardRow] = Field(default_factory=list)


class CompareSeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    value: float
    count: int = 0


class CompareSeriesRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    model_id: str
    group_key: str | None = None
    metric: str
    points: list[CompareSeriesPoint] = Field(default_factory=list)


class CompareSeriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["simple", "matched", "grouped"] = "simple"
    metric: str
    rolling_days: int = 1
    compared_model_ids: list[str] = Field(default_factory=list)
    series: list[CompareSeriesRecord] = Field(default_factory=list)


class CompareHistogramBin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bin_start: float
    bin_end: float
    count: int = 0


class CompareHistogramSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    model_id: str
    group_key: str | None = None
    bins: list[CompareHistogramBin] = Field(default_factory=list)


class CompareHistogramResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["simple", "matched", "grouped"] = "simple"
    metric: str
    bins: int = 10
    compared_model_ids: list[str] = Field(default_factory=list)
    series: list[CompareHistogramSeries] = Field(default_factory=list)


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
