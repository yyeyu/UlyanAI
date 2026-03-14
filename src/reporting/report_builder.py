"""Build experiment reports in TXT and PDF formats."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.service.event_store import EventStore

_REPORT_LOOKBACK_DAYS = 36_500
_REPORT_TEXT_WIDTH = 112


def resolve_report_paths(artifacts_root: Path, experiment_id: str) -> tuple[Path, Path, Path]:
    experiment_id_clean = str(experiment_id or "").strip()
    if not experiment_id_clean:
        raise ValueError("experiment_id is required")
    base_dir = (Path(artifacts_root).resolve() / "reports").resolve()
    report_dir = (base_dir / experiment_id_clean).resolve()
    if report_dir != base_dir and base_dir not in report_dir.parents:
        raise ValueError("invalid experiment_id path")
    txt_path = report_dir / "report.txt"
    pdf_path = report_dir / "report.pdf"
    return report_dir, txt_path, pdf_path


def generate_experiment_report_artifacts(
    *,
    store: EventStore,
    artifacts_root: Path,
    experiment_id: str,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    experiment_id_clean = str(experiment_id or "").strip()
    if not experiment_id_clean:
        raise ValueError("experiment_id is required")
    cycle_id_clean = str(cycle_id).strip() if cycle_id else None
    if cycle_id_clean == "":
        cycle_id_clean = None

    models = _load_experiment_models(store=store, experiment_id=experiment_id_clean)
    if not models:
        raise ValueError(f"no models found for experiment_id: {experiment_id_clean}")

    model_ids = [str(item["model_id"]) for item in models]
    events = _load_events_for_models(store=store, model_ids=model_ids, cycle_id=cycle_id_clean)
    scoreboard = store.compare_scoreboard(
        days=_REPORT_LOOKBACK_DAYS,
        asset=None,
        horizon=None,
        cycle_id=cycle_id_clean,
        model_ids=model_ids,
        mode="simple",
        exclude_stale=False,
        baseline_model_id=None,
        columns=None,
    )
    cycle_summary = store.cycle_summary(cycle_id_clean) if cycle_id_clean else None

    generated_at = datetime.now(timezone.utc).isoformat()
    report_text = build_report_text(
        experiment_id=experiment_id_clean,
        cycle_id=cycle_id_clean,
        generated_at=generated_at,
        models=models,
        events=events,
        scoreboard=scoreboard,
        cycle_summary=cycle_summary,
    )

    report_dir, txt_path, pdf_path = resolve_report_paths(artifacts_root, experiment_id_clean)
    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(report_text, encoding="utf-8")
    _write_pdf_from_text(pdf_path, report_text)
    return {
        "experiment_id": experiment_id_clean,
        "cycle_id": cycle_id_clean,
        "generated_at": generated_at,
        "models_count": len(models),
        "events_count": len(events),
        "report_dir": str(report_dir),
        "report_txt_path": str(txt_path),
        "report_pdf_path": str(pdf_path),
    }


def _load_experiment_models(*, store: EventStore, experiment_id: str) -> list[dict[str, Any]]:
    candidates = store.list_models(
        statuses=["active", "archived", "deleted"],
        limit=1000,
    )
    return [item for item in candidates if str(item.get("experiment_id") or "").strip() == experiment_id]


def _load_events_for_models(
    *,
    store: EventStore,
    model_ids: list[str],
    cycle_id: str | None,
) -> list[dict[str, Any]]:
    if not model_ids:
        return []
    out: list[dict[str, Any]] = []
    page = 1
    page_size = 500
    while True:
        payload = store.list_events(
            cycle_id=cycle_id,
            model_ids=model_ids,
            page=page,
            page_size=page_size,
            sort_by="created_at",
            sort_dir="asc",
        )
        items = list(payload.get("items", []))
        if not items:
            break
        out.extend(items)
        if len(out) >= int(payload.get("total", len(out))):
            break
        page += 1
    return out


def _fmt_num(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer():
        return str(int(num))
    return f"{num:.{precision}f}".rstrip("0").rstrip(".")


def _pick_numeric_metrics(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    keys_priority = [
        "coverage",
        "coverage_val",
        "coverage_test",
        "avg_width_pct",
        "width_pct",
        "avg_abs_error",
        "mae",
        "rmse",
        "pinball",
        "interval_score",
        "wis",
    ]
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key in keys_priority:
        if key in metrics and isinstance(metrics.get(key), (int, float)):
            out.append((key, float(metrics[key])))
            seen.add(key)
    for key, value in metrics.items():
        if key in seen:
            continue
        if isinstance(value, (int, float)):
            out.append((str(key), float(value)))
    return out[:10]


def _safe_scale_grid(values: Any) -> str:
    if not isinstance(values, list):
        return "-"
    if not values:
        return "[]"
    shown = ", ".join(_fmt_num(item, precision=4) for item in values[:8])
    suffix = ", ..." if len(values) > 8 else ""
    return f"[{shown}{suffix}]"


def _event_status_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"active": 0, "completed": 0, "cancelled": 0}
    for event in events:
        status = str(event.get("status") or "").strip().lower()
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def _recommendation_from_scoreboard(scoreboard_items: list[dict[str, Any]]) -> str:
    if not scoreboard_items:
        return "Insufficient event data to recommend a winner."
    best = min(
        scoreboard_items,
        key=lambda item: (
            float(item.get("avg_abs_error")) if item.get("avg_abs_error") is not None else float("inf"),
            float(item.get("avg_width_pct")) if item.get("avg_width_pct") is not None else float("inf"),
            -(float(item.get("coverage")) if item.get("coverage") is not None else -1.0),
            str(item.get("model_id") or ""),
        ),
    )
    model_id = str(best.get("model_id") or "-")
    coverage = _fmt_num(best.get("coverage"), precision=4)
    avg_abs_error = _fmt_num(best.get("avg_abs_error"), precision=6)
    return f"Primary candidate: {model_id} (coverage={coverage}, avg_abs_error={avg_abs_error})."


def _wrap_lines(lines: list[str], width: int = _REPORT_TEXT_WIDTH) -> list[str]:
    out: list[str] = []
    for line in lines:
        raw = str(line or "")
        if not raw.strip():
            out.append("")
            continue
        if raw.startswith("- "):
            wrapped = textwrap.wrap(
                raw[2:],
                width=max(20, width - 2),
                initial_indent="- ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            out.extend(wrapped or [raw])
            continue
        wrapped = textwrap.wrap(
            raw,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.extend(wrapped or [raw])
    return out


def build_report_text(
    *,
    experiment_id: str,
    cycle_id: str | None,
    generated_at: str,
    models: list[dict[str, Any]],
    events: list[dict[str, Any]],
    scoreboard: dict[str, Any],
    cycle_summary: dict[str, Any] | None,
) -> str:
    scoreboard_items = list(scoreboard.get("items", []))
    event_counts = _event_status_counts(events)
    completed_events = [item for item in events if str(item.get("status")).lower() == "completed"]
    assets = sorted({str(item.get("asset") or "-") for item in models})
    horizons = sorted({str(item.get("horizon") or "-") for item in models})
    model_ids = [str(item.get("model_id") or "-") for item in models]

    risks: list[str] = []
    if not completed_events:
        risks.append("No completed events found for selected filters; event metrics are limited.")
    if len(models) >= 1000:
        risks.append("Model lookup reached current API limit (1000); some models may be missing.")
    if any(not isinstance(item.get("metrics_json"), dict) or not item.get("metrics_json") for item in models):
        risks.append("Some models have empty offline metrics in the registry.")
    if len(assets) > 1:
        risks.append("Experiment contains multiple assets; comparisons may not be strictly homogeneous.")
    if len(horizons) > 1:
        risks.append("Experiment contains multiple horizons; aggregated metrics should be interpreted carefully.")
    if not risks:
        risks.append("No major data-quality issues detected in available metadata.")

    recommendation = _recommendation_from_scoreboard(scoreboard_items)
    if cycle_summary:
        summary_rec = str(cycle_summary.get("recommendation") or "").strip()
        summary_winner = str(cycle_summary.get("winner_model_id") or "").strip()
        if summary_rec:
            recommendation = f"Cycle recommendation: {summary_rec}" + (f", winner={summary_winner}." if summary_winner else ".")

    lines: list[str] = []
    lines.append("1. Executive Summary")
    lines.append(f"- report_generated_at_utc: {generated_at}")
    lines.append(f"- experiment_id: {experiment_id}")
    lines.append(f"- cycle_id: {cycle_id or '-'}")
    lines.append(f"- models_total: {len(models)}")
    lines.append(f"- events_total: {len(events)}")
    lines.append(f"- events_completed: {event_counts.get('completed', 0)}")
    lines.append(f"- events_active: {event_counts.get('active', 0)}")
    lines.append(f"- events_cancelled: {event_counts.get('cancelled', 0)}")
    lines.append("")
    lines.append("2. Scope and Inputs")
    lines.append(f"- assets: {', '.join(assets) if assets else '-'}")
    lines.append(f"- horizons: {', '.join(horizons) if horizons else '-'}")
    lines.append(f"- compared_model_ids_count: {len(scoreboard.get('compared_model_ids', []))}")
    lines.append(f"- report_mode: experiment_id" + (" + cycle_id filter" if cycle_id else ""))
    lines.append("")
    lines.append("3. Model Inventory")
    lines.append("model_id | asset | horizon | status | created_at | is_production")
    for model in models:
        lines.append(
            " | ".join(
                [
                    str(model.get("model_id") or "-"),
                    str(model.get("asset") or "-"),
                    str(model.get("horizon") or "-"),
                    str(model.get("status") or "-"),
                    str(model.get("created_at") or "-"),
                    "1" if bool(model.get("is_production")) else "0",
                ]
            )
        )
    lines.append("")
    lines.append("4. Training and Hyperparameters")
    for model in models:
        params = model.get("params_json") if isinstance(model.get("params_json"), dict) else {}
        hyper = params.get("hyperparams") if isinstance(params.get("hyperparams"), dict) else {}
        lines.append(f"- model_id: {model.get('model_id')}")
        lines.append(
            f"  base_timeframe={params.get('base_timeframe', '-')}, "
            f"steps_ahead={_fmt_num(params.get('steps_ahead'))}, "
            f"quantiles_count={len(params.get('quantiles', [])) if isinstance(params.get('quantiles'), list) else '-'}"
        )
        if hyper:
            hyper_text = ", ".join(f"{key}={_fmt_num(value, precision=6)}" for key, value in sorted(hyper.items()))
            lines.append(f"  hyperparams: {hyper_text}")
        else:
            lines.append("  hyperparams: -")
    lines.append("")
    lines.append("5. Calibration Configuration")
    for model in models:
        params = model.get("params_json") if isinstance(model.get("params_json"), dict) else {}
        calibrator = params.get("calibrator") if isinstance(params.get("calibrator"), dict) else {}
        lines.append(f"- model_id: {model.get('model_id')}")
        lines.append(
            "  "
            + ", ".join(
                [
                    f"method={params.get('calibration_method', '-')}",
                    f"scale_selection_rule={params.get('scale_selection_rule', '-')}",
                    f"scale_grid={_safe_scale_grid(params.get('scale_grid'))}",
                    f"chosen_scale={_fmt_num(calibrator.get('chosen_scale'), precision=6)}",
                ]
            )
        )
    lines.append("")
    lines.append("6. Offline Registry Metrics")
    for model in models:
        metrics = model.get("metrics_json") if isinstance(model.get("metrics_json"), dict) else {}
        picked = _pick_numeric_metrics(metrics)
        if picked:
            metrics_line = ", ".join(f"{key}={_fmt_num(value, precision=6)}" for key, value in picked)
        else:
            metrics_line = "-"
        lines.append(f"- {model.get('model_id')}: {metrics_line}")
    lines.append("")
    lines.append("7. Event Metrics and Leaderboard")
    lines.append("model_id | completed | coverage | avg_width_pct | avg_abs_error | avg_interval_score | avg_wis")
    for row in scoreboard_items:
        lines.append(
            " | ".join(
                [
                    str(row.get("model_id") or "-"),
                    _fmt_num(row.get("completed_events"), precision=0),
                    _fmt_num(row.get("coverage"), precision=4),
                    _fmt_num(row.get("avg_width_pct"), precision=6),
                    _fmt_num(row.get("avg_abs_error"), precision=6),
                    _fmt_num(row.get("avg_interval_score"), precision=6),
                    _fmt_num(row.get("avg_wis"), precision=6),
                ]
            )
        )
    if not scoreboard_items:
        lines.append("- Leaderboard is empty for selected filters.")
    lines.append("")
    lines.append("8. Cycle or Tournament Assessment")
    if cycle_summary:
        lines.append(f"- cycle_status: {cycle_summary.get('status')}")
        lines.append(f"- recommendation: {cycle_summary.get('recommendation')}")
        lines.append(f"- winner_model_id: {cycle_summary.get('winner_model_id') or '-'}")
        lines.append(f"- paired_runs: {cycle_summary.get('paired_runs', 0)}")
        lines.append(f"- early_stop_triggered: {cycle_summary.get('early_stop_triggered', False)}")
    else:
        lines.append("- cycle_id not provided; cycle-level assessment skipped.")
    lines.append("")
    lines.append("9. Risks and Observations")
    for risk in risks:
        lines.append(f"- {risk}")
    lines.append("")
    lines.append("10. Recommendation and Appendix")
    lines.append(f"- recommendation: {recommendation}")
    lines.append(f"- appendix_model_ids: {', '.join(model_ids)}")
    lines.append(f"- appendix_scoreboard_rows: {len(scoreboard_items)}")
    lines.append(f"- appendix_completed_events: {len(completed_events)}")

    wrapped_lines = _wrap_lines(lines)
    return ("\n".join(wrapped_lines)).rstrip() + "\n"


def _write_pdf_from_text(pdf_path: Path, content: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - dependency presence is environment-specific
        raise RuntimeError("reportlab is required for PDF report generation") from exc

    margin_x = 36
    margin_y = 36
    line_height = 12
    font_name = "Courier"
    font_size = 9

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    report_canvas = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    y = height - margin_y
    report_canvas.setFont(font_name, font_size)

    for line in content.splitlines():
        if y <= margin_y:
            report_canvas.showPage()
            report_canvas.setFont(font_name, font_size)
            y = height - margin_y
        report_canvas.drawString(margin_x, y, line)
        y -= line_height
    report_canvas.save()
