"""Evaluation summaries focused on price intervals for product consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.utils import dump_json


@dataclass(frozen=True)
class PriceIntervalMetrics:
    coverage: float
    mean_width: float
    mean_relative_width: float
    sample_size: int


def compute_price_interval_metrics(
    actual_prices: Sequence[float],
    interval_lows: Sequence[float],
    interval_highs: Sequence[float],
) -> PriceIntervalMetrics:
    if not (len(actual_prices) == len(interval_lows) == len(interval_highs)):
        raise ValueError("all input sequences must have the same length")
    if not actual_prices:
        raise ValueError("at least one sample is required")

    n = len(actual_prices)
    covered = 0
    widths: list[float] = []
    rel_widths: list[float] = []

    for actual, low, high in zip(actual_prices, interval_lows, interval_highs):
        if low > high:
            raise ValueError("interval low cannot exceed interval high")
        if low <= actual <= high:
            covered += 1
        width = high - low
        widths.append(width)
        scale = max(abs(actual), 1e-12)
        rel_widths.append(width / scale)

    return PriceIntervalMetrics(
        coverage=covered / n,
        mean_width=sum(widths) / n,
        mean_relative_width=sum(rel_widths) / n,
        sample_size=n,
    )


def build_price_report(metrics_by_horizon: Mapping[str, PriceIntervalMetrics]) -> str:
    lines = [
        "# Price-Range Quality Report",
        "",
        "| Horizon | Coverage | Mean Width | Mean Relative Width | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon, metrics in metrics_by_horizon.items():
        lines.append(
            f"| {horizon} | {metrics.coverage:.4f} | {metrics.mean_width:.4f} | "
            f"{metrics.mean_relative_width:.4f} | {metrics.sample_size} |"
        )
    return "\n".join(lines)


def build_walk_forward_markdown(title: str, walk_forward_result: dict) -> str:
    lines = [f"# {title}", ""]
    summary = walk_forward_result.get("summary", {})
    splits = walk_forward_result.get("splits", [])

    if summary:
        lines.extend(
            [
                "## Summary",
                "",
                f"- splits_count: {summary.get('splits_count', 0)}",
                f"- avg_pinball_q10: {summary.get('avg_pinball_q10', float('nan')):.6f}",
                f"- avg_pinball_q50: {summary.get('avg_pinball_q50', float('nan')):.6f}",
                f"- avg_pinball_q90: {summary.get('avg_pinball_q90', float('nan')):.6f}",
                f"- avg_coverage: {summary.get('avg_coverage', float('nan')):.6f}",
                f"- avg_mean_width: {summary.get('avg_mean_width', float('nan')):.6f}",
                "",
            ]
        )

    if splits:
        lines.extend(
            [
                "## Splits",
                "",
                "| Split | Coverage | Mean Width | Pinball q10 | Pinball q50 | Pinball q90 | Scale | Test Start | Test End |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for split in splits:
            lines.append(
                "| {split_index} | {coverage:.4f} | {mean_width:.4f} | {pinball_q10:.6f} | "
                "{pinball_q50:.6f} | {pinball_q90:.6f} | {calibrator_scale:.3f} | {test_period_start} | {test_period_end} |".format(
                    **split
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_walk_forward_report(report_dir: Path, name: str, result: dict) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"{name}.md"
    json_path = report_dir / f"{name}.json"
    markdown = build_walk_forward_markdown(title=f"Walk-forward report: {name}", walk_forward_result=result)
    md_path.write_text(markdown, encoding="utf-8")
    dump_json(json_path, result)
    return md_path, json_path
