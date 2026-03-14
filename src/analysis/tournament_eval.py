"""Tournament-level paired evaluation and early-stop recommendation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Any

DEFAULT_TOURNAMENT_RULES: dict[str, float] = {
    "min_paired_runs": 5.0,
    "win_rate_threshold": 0.65,
    "win_rate_confidence_z": 1.96,
    "delta_width_min_runs": 4.0,
    "delta_width_ci_max": 0.20,
}


def _to_iso_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN guard
        return None
    return numeric


def _normalize_rules(raw_rules: dict[str, Any] | None) -> dict[str, float]:
    rules = dict(DEFAULT_TOURNAMENT_RULES)
    if not isinstance(raw_rules, dict):
        return rules
    for key in DEFAULT_TOURNAMENT_RULES:
        candidate = _to_float(raw_rules.get(key))
        if candidate is None:
            continue
        if key in {"min_paired_runs", "delta_width_min_runs"}:
            rules[key] = float(max(1, int(candidate)))
        elif key in {"win_rate_threshold"}:
            rules[key] = max(0.0, min(1.0, candidate))
        elif key in {"win_rate_confidence_z", "delta_width_ci_max"}:
            rules[key] = max(0.0, candidate)
        else:
            rules[key] = candidate
    return rules


def _wilson_interval(successes: int, total: int, z_score: float) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    phat = max(0.0, min(1.0, float(successes) / float(total)))
    z2 = z_score * z_score
    denominator = 1.0 + z2 / total
    center = phat + z2 / (2.0 * total)
    margin = z_score * sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return max(0.0, lower), min(1.0, upper)


def _mean_ci(values: list[float], z_score: float) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_half_width": None, "lower": None, "upper": None}
    count = len(values)
    mean_value = sum(values) / float(count)
    if count <= 1:
        return {
            "mean": mean_value,
            "ci_half_width": None,
            "lower": None,
            "upper": None,
        }
    variance = sum((value - mean_value) ** 2 for value in values) / float(count - 1)
    std = sqrt(max(0.0, variance))
    ci_half_width = z_score * (std / sqrt(float(count)))
    return {
        "mean": mean_value,
        "ci_half_width": ci_half_width,
        "lower": mean_value - ci_half_width,
        "upper": mean_value + ci_half_width,
    }


def _rank_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
        hit = row.get("hit")
        hit_rank = 0.0 if isinstance(hit, bool) and hit else 1.0
        abs_error = _to_float(row.get("abs_error"))
        width_pct = _to_float(row.get("width_pct"))
        return (
            hit_rank,
            abs_error if abs_error is not None else float("inf"),
            width_pct if width_pct is not None else float("inf"),
            str(row.get("model_id", "")),
        )

    return sorted(rows, key=_sort_key)


def _normalize_cycle_model_ids(cycle: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    raw_cycle_ids = list(cycle.get("model_ids", []))
    if cycle.get("model_id"):
        raw_cycle_ids.append(str(cycle["model_id"]))
    for raw in raw_cycle_ids:
        model_id = str(raw).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)
    for event in events:
        model_id = str(event.get("model_id", "")).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)
    return ordered


def build_tournament_summary(
    *,
    cycle: dict[str, Any],
    events: list[dict[str, Any]],
    model_lookup: dict[str, dict[str, Any]] | None = None,
    rules_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = _normalize_rules(rules_override)
    lookup = model_lookup or {}
    cycle_model_ids = _normalize_cycle_model_ids(cycle, events)
    cycle_model_set = set(cycle_model_ids)

    run_rows: dict[int, dict[str, dict[str, Any]]] = {}
    for event in events:
        model_id = str(event.get("model_id", "")).strip()
        if not model_id:
            continue
        cycle_seq_raw = event.get("cycle_seq")
        try:
            cycle_seq = int(cycle_seq_raw or 0)
        except (TypeError, ValueError):
            continue
        if cycle_seq <= 0:
            continue
        metrics = event.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        row = {
            "event_id": str(event.get("event_id", "")),
            "model_id": model_id,
            "status": str(event.get("status", "")).strip().lower(),
            "created_at": str(event.get("created_at", "")),
            "updated_at": str(event.get("updated_at", "")),
            "eval_key": str(event.get("eval_key", "")),
            "hit": metrics.get("hit") if isinstance(metrics.get("hit"), bool) else None,
            "abs_error": _to_float(metrics.get("abs_error")),
            "width_pct": _to_float(metrics.get("width_pct")),
        }
        bucket = run_rows.setdefault(cycle_seq, {})
        prev = bucket.get(model_id)
        if prev and str(prev.get("updated_at", "")) > row["updated_at"]:
            continue
        bucket[model_id] = row

    model_stats: dict[str, dict[str, Any]] = {
        model_id: {
            "model_id": model_id,
            "total_events": 0,
            "completed_events": 0,
            "hits": 0,
            "abs_sum": 0.0,
            "abs_count": 0,
            "width_sum": 0.0,
            "width_count": 0,
            "paired_runs": 0,
            "wins": 0,
        }
        for model_id in cycle_model_ids
    }

    paired_metrics: list[dict[str, Any]] = []
    paired_closed_runs = 0
    for cycle_seq in sorted(run_rows.keys()):
        run_map = run_rows[cycle_seq]
        present_model_ids = sorted(run_map.keys())
        required_models = cycle_model_set or set(present_model_ids)
        pair_ready = required_models.issubset(set(present_model_ids)) if required_models else False
        run_closed = pair_ready and all(
            str(run_map.get(model_id, {}).get("status", "")) in {"completed", "cancelled"}
            for model_id in required_models
        )

        completed_models = 0
        cancelled_models = 0
        active_models = 0
        for model_id in cycle_model_ids:
            row = run_map.get(model_id)
            if row is None:
                continue
            stat = model_stats.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "total_events": 0,
                    "completed_events": 0,
                    "hits": 0,
                    "abs_sum": 0.0,
                    "abs_count": 0,
                    "width_sum": 0.0,
                    "width_count": 0,
                    "paired_runs": 0,
                    "wins": 0,
                },
            )
            stat["total_events"] += 1
            status = str(row.get("status", ""))
            if status == "completed":
                completed_models += 1
                stat["completed_events"] += 1
                if isinstance(row.get("hit"), bool) and bool(row["hit"]):
                    stat["hits"] += 1
                if row.get("abs_error") is not None:
                    stat["abs_sum"] += float(row["abs_error"])
                    stat["abs_count"] += 1
                if row.get("width_pct") is not None:
                    stat["width_sum"] += float(row["width_pct"])
                    stat["width_count"] += 1
            elif status == "cancelled":
                cancelled_models += 1
            elif status == "active":
                active_models += 1

        if run_closed:
            for model_id in cycle_model_ids:
                row = run_map.get(model_id)
                if row is not None and str(row.get("status", "")) == "completed":
                    model_stats[model_id]["paired_runs"] += 1

        rank_candidates = [
            run_map[model_id]
            for model_id in cycle_model_ids
            if model_id in run_map and str(run_map[model_id].get("status", "")) == "completed"
        ]
        winner: dict[str, Any] | None = None
        runner_up: dict[str, Any] | None = None
        if run_closed and len(rank_candidates) >= 2:
            ranked_rows = _rank_run_rows(rank_candidates)
            winner = ranked_rows[0]
            runner_up = ranked_rows[1]
            model_stats[str(winner["model_id"])]["wins"] += 1
            paired_closed_runs += 1

        paired_metrics.append(
            {
                "cycle_seq": int(cycle_seq),
                "created_at": max((str(item.get("created_at", "")) for item in run_map.values()), default=None),
                "eval_key": next(
                    (str(item.get("eval_key", "")) for item in run_map.values() if str(item.get("eval_key", ""))),
                    None,
                ),
                "models_present": present_model_ids,
                "pair_ready": bool(pair_ready),
                "run_closed": bool(run_closed),
                "completed_models": int(completed_models),
                "cancelled_models": int(cancelled_models),
                "active_models": int(active_models),
                "winner_model_id": None if winner is None else str(winner["model_id"]),
                "runner_up_model_id": None if runner_up is None else str(runner_up["model_id"]),
                "winner_abs_error": None if winner is None or winner.get("abs_error") is None else float(winner["abs_error"]),
                "runner_up_abs_error": (
                    None if runner_up is None or runner_up.get("abs_error") is None else float(runner_up["abs_error"])
                ),
                "delta_abs_error": (
                    None
                    if winner is None
                    or runner_up is None
                    or winner.get("abs_error") is None
                    or runner_up.get("abs_error") is None
                    else float(runner_up["abs_error"]) - float(winner["abs_error"])
                ),
                "delta_width_pct": (
                    None
                    if winner is None
                    or runner_up is None
                    or winner.get("width_pct") is None
                    or runner_up.get("width_pct") is None
                    else float(runner_up["width_pct"]) - float(winner["width_pct"])
                ),
            }
        )

    leaderboard: list[dict[str, Any]] = []
    for model_id in cycle_model_ids:
        stat = model_stats[model_id]
        completed_events = int(stat["completed_events"])
        paired_runs = int(stat["paired_runs"])
        wins = int(stat["wins"])
        model_meta = lookup.get(model_id, {})
        leaderboard.append(
            {
                "rank": 0,
                "model_id": model_id,
                "model_name": model_meta.get("name"),
                "total_events": int(stat["total_events"]),
                "completed_events": completed_events,
                "paired_runs": paired_runs,
                "wins": wins,
                "win_rate": (wins / float(paired_runs)) if paired_runs > 0 else None,
                "coverage": (stat["hits"] / float(completed_events)) if completed_events > 0 else None,
                "avg_abs_error": (stat["abs_sum"] / float(stat["abs_count"])) if stat["abs_count"] > 0 else None,
                "avg_width_pct": (stat["width_sum"] / float(stat["width_count"])) if stat["width_count"] > 0 else None,
            }
        )
    leaderboard.sort(
        key=lambda item: (
            -int(item.get("wins", 0)),
            -(float(item.get("win_rate")) if item.get("win_rate") is not None else -1.0),
            float(item.get("avg_abs_error")) if item.get("avg_abs_error") is not None else float("inf"),
            str(item.get("model_id", "")),
        )
    )
    for index, item in enumerate(leaderboard, start=1):
        item["rank"] = index

    winner_model_id: str | None = str(leaderboard[0]["model_id"]) if leaderboard else None
    runner_model_id: str | None = str(leaderboard[1]["model_id"]) if len(leaderboard) > 1 else None
    top = leaderboard[0] if leaderboard else None

    win_rate_lb = None
    win_rate_ub = None
    win_rate_stop = False
    if top is not None:
        top_wins = int(top.get("wins", 0))
        top_paired = int(top.get("paired_runs", 0))
        win_rate_lb, win_rate_ub = _wilson_interval(
            top_wins,
            top_paired,
            float(rules["win_rate_confidence_z"]),
        )
        if top_paired >= int(rules["min_paired_runs"]) and win_rate_lb is not None:
            win_rate_stop = bool(win_rate_lb >= float(rules["win_rate_threshold"]))

    delta_width_values: list[float] = []
    if winner_model_id and runner_model_id:
        for run in paired_metrics:
            if not bool(run.get("run_closed", False)):
                continue
            cycle_seq = int(run.get("cycle_seq", 0))
            model_map = run_rows.get(cycle_seq, {})
            top_row = model_map.get(winner_model_id)
            runner_row = model_map.get(runner_model_id)
            if top_row is None or runner_row is None:
                continue
            if str(top_row.get("status", "")) != "completed" or str(runner_row.get("status", "")) != "completed":
                continue
            top_width = _to_float(top_row.get("width_pct"))
            runner_width = _to_float(runner_row.get("width_pct"))
            if top_width is None or runner_width is None:
                continue
            delta_width_values.append(top_width - runner_width)

    delta_width_ci = _mean_ci(delta_width_values, float(rules["win_rate_confidence_z"]))
    delta_width_stop = bool(
        len(delta_width_values) >= int(rules["delta_width_min_runs"])
        and delta_width_ci.get("ci_half_width") is not None
        and float(delta_width_ci["ci_half_width"]) <= float(rules["delta_width_ci_max"])
        and delta_width_ci.get("upper") is not None
        and float(delta_width_ci["upper"]) <= 0.0
    )

    stop_reasons: list[str] = []
    if win_rate_stop and win_rate_lb is not None:
        stop_reasons.append(
            f"win_rate_lb={win_rate_lb:.3f} >= {float(rules['win_rate_threshold']):.3f}"
        )
    if delta_width_stop and delta_width_ci.get("upper") is not None:
        stop_reasons.append(
            f"delta_width_upper_ci={float(delta_width_ci['upper']):.4f} <= 0"
        )

    early_stop_triggered = bool(stop_reasons)
    cycle_status = str(cycle.get("status", "running")).strip().lower()
    if cycle_status == "running":
        if early_stop_triggered:
            recommendation = "winner" if winner_model_id else "stop"
        else:
            recommendation = "continue"
    else:
        recommendation = "winner" if winner_model_id else "stop"

    launched_runs = max(0, int(cycle.get("launched_runs", 0) or 0))
    completed_runs = max(0, int(cycle.get("completed_runs", 0) or 0))
    cancelled_runs = max(0, int(cycle.get("cancelled_runs", 0) or 0))
    total_runs = max(0, int(cycle.get("total_runs", 0) or 0))
    active_runs = max(0, launched_runs - completed_runs - cancelled_runs)
    remaining_runs = max(0, total_runs - launched_runs)

    return {
        "cycle_id": str(cycle.get("cycle_id", "")),
        "asset": str(cycle.get("asset", "")),
        "horizon": str(cycle.get("horizon", "")),
        "mode": str(cycle.get("mode", "single")),
        "status": str(cycle.get("status", "running")),
        "total_runs": total_runs,
        "launched_runs": launched_runs,
        "completed_runs": completed_runs,
        "cancelled_runs": cancelled_runs,
        "observed_runs": len(run_rows),
        "paired_runs": paired_closed_runs,
        "active_runs": active_runs,
        "remaining_runs": remaining_runs,
        "compared_model_ids": cycle_model_ids,
        "recommendation": recommendation,
        "winner_model_id": winner_model_id,
        "stop_reason": stop_reasons[0] if stop_reasons else None,
        "early_stop_triggered": early_stop_triggered,
        "rules": {
            "min_paired_runs": int(rules["min_paired_runs"]),
            "win_rate_threshold": float(rules["win_rate_threshold"]),
            "win_rate_confidence_z": float(rules["win_rate_confidence_z"]),
            "delta_width_min_runs": int(rules["delta_width_min_runs"]),
            "delta_width_ci_max": float(rules["delta_width_ci_max"]),
        },
        "early_stop_checks": {
            "win_rate_lb": win_rate_lb,
            "win_rate_ub": win_rate_ub,
            "win_rate_stop": win_rate_stop,
            "delta_width_samples": len(delta_width_values),
            "delta_width_mean": delta_width_ci.get("mean"),
            "delta_width_ci_half_width": delta_width_ci.get("ci_half_width"),
            "delta_width_ci_lower": delta_width_ci.get("lower"),
            "delta_width_ci_upper": delta_width_ci.get("upper"),
            "delta_width_stop": delta_width_stop,
            "reasons": stop_reasons,
        },
        "leaderboard": leaderboard,
        "paired_metrics": paired_metrics,
        "generated_at": _to_iso_utc(datetime.now(timezone.utc)),
    }
