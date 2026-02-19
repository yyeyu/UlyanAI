"""Paper simulation utilities using price ranges as the public interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SimulationParams:
    edge_threshold: float = 0.05
    fees: float = 0.02
    max_position_per_market: float = 1.0
    max_daily_loss: float = 1.0
    max_total_exposure: float = 3.0


def overlap_probability(
    *,
    forecast_low: float,
    forecast_high: float,
    market_low: float,
    market_high: float,
) -> float:
    if forecast_low > forecast_high:
        raise ValueError("forecast_low must be <= forecast_high")
    if market_low >= market_high:
        raise ValueError("market_low must be < market_high")

    overlap = max(0.0, min(forecast_high, market_high) - max(forecast_low, market_low))
    market_width = market_high - market_low
    return min(1.0, max(0.0, overlap / market_width))


def should_enter_trade(p_model: float, p_market: float, edge_threshold: float) -> bool:
    return p_model >= p_market + edge_threshold


def run_paper_sim(
    rows: Iterable[dict[str, float]],
    params: SimulationParams | None = None,
) -> dict[str, Any]:
    cfg = params or SimulationParams()
    pnl_cum = 0.0
    daily_pnl = 0.0
    exposure = 0.0
    trade_logs: list[dict[str, Any]] = []
    entered = 0

    for row in rows:
        p_model = overlap_probability(
            forecast_low=row["forecast_low"],
            forecast_high=row["forecast_high"],
            market_low=row["market_low"],
            market_high=row["market_high"],
        )
        p_market = row["market_price"]
        edge = p_model - p_market
        gross_ev = edge
        net_ev = gross_ev - cfg.fees
        enter = should_enter_trade(p_model, p_market, cfg.edge_threshold)
        position_size = min(float(row.get("position_size", 1.0)), cfg.max_position_per_market)

        if exposure + position_size > cfg.max_total_exposure:
            enter = False
        if daily_pnl <= -cfg.max_daily_loss:
            enter = False

        if enter:
            entered += 1
            exposure += position_size
            trade_pnl = net_ev * position_size
            pnl_cum += trade_pnl
            daily_pnl += trade_pnl
        else:
            trade_pnl = 0.0

        trade_logs.append(
            {
                "asset": row.get("asset", "BTC"),
                "horizon": row.get("horizon", "1h"),
                "forecast_price_range": [row["forecast_low"], row["forecast_high"]],
                "market_price_range": [row["market_low"], row["market_high"]],
                "p_model": p_model,
                "p_market": p_market,
                "edge": edge,
                "gross_ev": gross_ev,
                "fees": cfg.fees,
                "net_ev": net_ev,
                "enter_trade": enter,
                "position_size": position_size,
                "trade_pnl": trade_pnl,
            }
        )

    total = len(trade_logs)
    avg_edge = sum(item["edge"] for item in trade_logs) / total if total else 0.0
    return {
        "summary": {
            "trades_total": total,
            "trades_entered": entered,
            "pnl_cumulative": pnl_cum,
            "avg_edge": avg_edge,
            "final_exposure": exposure,
            "mode": "price_ranges",
        },
        "trades": trade_logs,
    }
