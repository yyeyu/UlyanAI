"""End-to-end project pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import get_runtime_paths
from src.data.clean import clean_ohlcv
from src.data.download import DownloadParams, download_ohlcv
from src.data.qc import assert_qc, run_qc
from src.data.resample import resample_ohlcv
from src.eval.report import write_walk_forward_report
from src.eval.walk_forward import run_walk_forward
from src.features.build import build_features, finalize_features
from src.labels.build import build_labels, join_features_labels
from src.models.registry import latest_model_dir
from src.models.train import train_horizon_model
from src.sim.polymarket import SimulationParams, run_paper_sim
from src.utils import (
    dump_json,
    read_all_parquet,
    utc_now,
    write_partitioned_yearly_parquet,
)


def _symbol_info(config: dict[str, Any], asset: str) -> dict[str, Any]:
    assets = config.get("assets", {})
    if asset.upper() not in assets:
        raise ValueError(f"asset={asset} not found in configs/assets.yaml")
    return assets[asset.upper()]


def _parquet_folder(data_root: Path, layer: str, exchange: str, symbol: str, key: str) -> Path:
    symbol_dir = symbol.replace("/", "_")
    return data_root / layer / exchange / symbol_dir / key


def _to_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _to_rel_paths(paths: list[Path], root: Path) -> list[str]:
    return [_to_rel_path(path, root) for path in paths]


def _asset_price_anchor(asset: str) -> float:
    anchors = {
        "BTC": 50000.0,
        "ETH": 3000.0,
        "SOL": 120.0,
    }
    return float(anchors.get(asset.upper(), 1000.0))


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _hash_payload(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "empty"
    out = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    if "ts_utc" in out.columns:
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce").astype("int64")
    row_hashes = pd.util.hash_pandas_object(out, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def _build_run_meta(*, config: dict[str, Any], root: Path, dataset_hash: str) -> dict[str, str]:
    return {
        "generated_at_utc": utc_now().isoformat(),
        "git_commit": _git_commit(root),
        "config_hash": _hash_payload(config),
        "dataset_hash": dataset_hash,
    }


def run_data_pipeline(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    symbol_info = _symbol_info(config, asset)
    symbol = symbol_info["symbol"]
    market_symbol = symbol_info["market_symbol"]
    exchange = config.get("exchange", "binance")
    base_tf = config.get("base_timeframe", "1m")

    days = int(config.get("data", {}).get("download_days", 120))
    end = utc_now()
    start = end - timedelta(days=days)
    download_df = download_ohlcv(
        DownloadParams(
            symbol_market=market_symbol,
            interval=base_tf,
            start=start,
            end=end,
            limit=int(config.get("data", {}).get("ohlcv_limit_per_call", 1000)),
            offline_fallback=bool(config.get("data", {}).get("allow_offline_synthetic_fallback", True)),
        )
    )
    raw_written = write_partitioned_yearly_parquet(
        download_df,
        data_root=paths.data_root,
        layer="raw",
        exchange=exchange,
        symbol=symbol,
        timeframe=base_tf,
        compression=config.get("storage", {}).get("parquet_compression", "snappy"),
    )

    clean_df = clean_ohlcv(download_df, symbol=symbol)
    assert_qc(
        clean_df,
        timeframe=base_tf,
        max_missing_ratio=float(config.get("qc", {}).get("max_missing_ratio", 0.001)),
    )
    clean_written = write_partitioned_yearly_parquet(
        clean_df,
        data_root=paths.data_root,
        layer="clean",
        exchange=exchange,
        symbol=symbol,
        timeframe=base_tf,
        compression=config.get("storage", {}).get("parquet_compression", "snappy"),
    )

    resampled_qc: dict[str, Any] = {}
    resampled_counts: dict[str, int] = {}
    for tf in config.get("timeframes", []):
        resampled = resample_ohlcv(clean_df, tf)
        qc_result = run_qc(
            resampled,
            timeframe=tf,
            max_missing_ratio=float(config.get("qc", {}).get("max_missing_ratio", 0.001)),
        )
        if not qc_result.ok:
            failed = [k for k, v in qc_result.checks.items() if not v]
            raise ValueError(f"Resampled QC failed for {tf}: {failed} -> {qc_result.details}")

        write_partitioned_yearly_parquet(
            resampled,
            data_root=paths.data_root,
            layer="resampled",
            exchange=exchange,
            symbol=symbol,
            timeframe=tf,
            compression=config.get("storage", {}).get("parquet_compression", "snappy"),
        )
        resampled_qc[tf] = qc_result.details
        resampled_counts[tf] = int(len(resampled))

    dataset_hash = _hash_dataframe(
        clean_df,
        columns=["ts_utc", "open", "high", "low", "close", "volume", "symbol", "exchange"],
    )
    payload = {
        "asset": asset.upper(),
        "symbol": symbol,
        "exchange": exchange,
        "base_timeframe": base_tf,
        "download_start": start.isoformat(),
        "download_end": end.isoformat(),
        "raw_files": _to_rel_paths(raw_written, paths.root),
        "clean_files": _to_rel_paths(clean_written, paths.root),
        "resampled_counts": resampled_counts,
        "resampled_qc": resampled_qc,
        "meta": _build_run_meta(config=config, root=paths.root, dataset_hash=dataset_hash),
    }
    dump_json(paths.artifacts_root / "runs" / f"data_{asset.lower()}.json", payload)
    return payload


def run_feature_label_pipeline(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    symbol_info = _symbol_info(config, asset)
    symbol = symbol_info["symbol"]
    exchange = config.get("exchange", "binance")

    feature_cfg = config.get("feature_set_config", {})
    outputs: dict[str, Any] = {}
    for horizon, horizon_info in config.get("horizons_map", {}).items():
        timeframe = horizon_info["timeframe"]
        src_dir = _parquet_folder(paths.data_root, "resampled", exchange, symbol, timeframe)
        candles = read_all_parquet(src_dir)
        if candles.empty:
            continue
        features = finalize_features(build_features(candles, feature_cfg), warmup_drop=True)
        labels = build_labels(candles, horizon_name=horizon, steps_ahead=1)

        feat_files = write_partitioned_yearly_parquet(
            features,
            data_root=paths.data_root,
            layer="features",
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            compression=config.get("storage", {}).get("parquet_compression", "snappy"),
        )
        lbl_files = write_partitioned_yearly_parquet(
            labels,
            data_root=paths.data_root,
            layer="labels",
            exchange=exchange,
            symbol=symbol,
            timeframe=horizon,
            compression=config.get("storage", {}).get("parquet_compression", "snappy"),
        )
        outputs[horizon] = {
            "timeframe": timeframe,
            "features_rows": int(len(features)),
            "labels_rows": int(len(labels)),
            "feature_files": _to_rel_paths(feat_files, paths.root),
            "label_files": _to_rel_paths(lbl_files, paths.root),
        }

    payload: dict[str, Any] = {
        "items": outputs,
        "meta": _build_run_meta(
            config=config,
            root=paths.root,
            dataset_hash=_hash_payload(outputs),
        ),
    }
    dump_json(paths.artifacts_root / "runs" / f"features_labels_{asset.lower()}.json", payload)
    return outputs


def run_training_pipeline(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    symbol_info = _symbol_info(config, asset)
    symbol = symbol_info["symbol"]
    exchange = config.get("exchange", "binance")

    training_results: dict[str, Any] = {}
    training_errors: dict[str, str] = {}
    for horizon, horizon_info in config.get("horizons_map", {}).items():
        timeframe = horizon_info["timeframe"]
        features_dir = _parquet_folder(paths.data_root, "features", exchange, symbol, timeframe)
        labels_dir = _parquet_folder(paths.data_root, "labels", exchange, symbol, horizon)
        features_df = read_all_parquet(features_dir)
        labels_df = read_all_parquet(labels_dir)
        if features_df.empty or labels_df.empty:
            continue

        try:
            result = train_horizon_model(
                asset=asset.upper(),
                horizon=horizon,
                features_df=features_df,
                labels_df=labels_df,
                config=config,
                artifacts_root=paths.artifacts_root,
            )
            if "artifact_dir" in result:
                result["artifact_dir"] = _to_rel_path(Path(str(result["artifact_dir"])), paths.root)
            training_results[horizon] = result
        except Exception as exc:
            training_errors[horizon] = str(exc)

    payload = {
        "trained": training_results,
        "errors": training_errors,
        "meta": _build_run_meta(
            config=config,
            root=paths.root,
            dataset_hash=_hash_payload({"trained": training_results, "errors": training_errors}),
        ),
    }
    dump_json(paths.artifacts_root / "runs" / f"training_{asset.lower()}.json", payload)
    return payload


def run_walk_forward_pipeline(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    symbol_info = _symbol_info(config, asset)
    symbol = symbol_info["symbol"]
    exchange = config.get("exchange", "binance")
    results: dict[str, Any] = {}

    for horizon, horizon_info in config.get("horizons_map", {}).items():
        timeframe = horizon_info["timeframe"]
        features_df = read_all_parquet(_parquet_folder(paths.data_root, "features", exchange, symbol, timeframe))
        labels_df = read_all_parquet(_parquet_folder(paths.data_root, "labels", exchange, symbol, horizon))
        if features_df.empty or labels_df.empty:
            continue

        label_col = f"target_r_{horizon}"
        dataset = join_features_labels(features_df, labels_df, label_col).dropna().reset_index(drop=True)
        if dataset.empty:
            continue
        feature_cols = [c for c in dataset.columns if c not in {"ts_utc", "symbol", "exchange", label_col}]
        wf_result = run_walk_forward(
            dataset,
            feature_cols=feature_cols,
            label_col=label_col,
            split_cfg=config.get("walk_forward", {}),
            seed=int(config.get("seed", 42)),
        )
        results[horizon] = wf_result
        write_walk_forward_report(
            report_dir=paths.artifacts_root / "reports" / asset.upper(),
            name=f"walk_forward_{horizon}",
            result=wf_result,
        )

    payload: dict[str, Any] = dict(results)
    payload["meta"] = _build_run_meta(
        config=config,
        root=paths.root,
        dataset_hash=_hash_payload(results),
    )
    dump_json(paths.artifacts_root / "runs" / f"walk_forward_{asset.lower()}.json", payload)
    return payload


def run_simulation_pipeline(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    sim_cfg = config.get("sim", {})
    rows = []

    # Synthetic market rows for paper simulation; in prod this should be fed by Polymarket quotes.
    rng = np.random.default_rng(seed=int(config.get("seed", 42)))
    center_anchor = _asset_price_anchor(asset)
    for horizon in config.get("horizons", []):
        for _ in range(40):
            center = float(rng.normal(center_anchor, max(center_anchor * 0.03, 1.0)))
            spread = float(abs(rng.normal(max(center_anchor * 0.01, 0.5), max(center_anchor * 0.003, 0.2))))
            forecast_low = center - spread
            forecast_high = center + spread
            market_low = center - spread * float(rng.uniform(0.8, 1.2))
            market_high = center + spread * float(rng.uniform(0.8, 1.2))
            rows.append(
                {
                    "asset": asset.upper(),
                    "horizon": horizon,
                    "forecast_low": min(forecast_low, forecast_high),
                    "forecast_high": max(forecast_low, forecast_high),
                    "market_low": min(market_low, market_high),
                    "market_high": max(market_low, market_high) + 1e-6,
                    "market_price": float(rng.uniform(0.2, 0.8)),
                }
            )

    sim_result = run_paper_sim(
        rows=rows,
        params=SimulationParams(
            edge_threshold=float(sim_cfg.get("edge_threshold", 0.05)),
            fees=float(sim_cfg.get("fees", 0.02)),
            max_position_per_market=float(sim_cfg.get("max_position_per_market", 1.0)),
            max_daily_loss=float(sim_cfg.get("max_daily_loss", 1.0)),
            max_total_exposure=float(sim_cfg.get("max_total_exposure", 3.0)),
        ),
    )
    payload = dict(sim_result)
    payload["meta"] = _build_run_meta(
        config=config,
        root=paths.root,
        dataset_hash=_hash_payload(sim_result),
    )
    dump_json(paths.artifacts_root / "runs" / f"sim_{asset.lower()}.json", payload)
    return {"summary": sim_result.get("summary", {}), "meta": payload["meta"]}


def run_doctor(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    paths = get_runtime_paths(config, root=root)
    asset_u = asset.upper()
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        _symbol_info(config, asset_u)
        add_check("config.asset", True, f"{asset_u} configured")
    except Exception as exc:
        add_check("config.asset", False, str(exc))

    horizons_map = config.get("horizons_map", {})
    add_check("config.horizons", bool(horizons_map), f"count={len(horizons_map)}")

    add_check("paths.data_root", paths.data_root.exists(), str(paths.data_root))
    add_check("paths.artifacts_root", paths.artifacts_root.exists(), str(paths.artifacts_root))

    model_checks = []
    for horizon in sorted(horizons_map.keys()):
        latest = latest_model_dir(paths.artifacts_root, asset_u, horizon)
        model_checks.append({"horizon": horizon, "model_dir": str(latest) if latest else None, "ok": latest is not None})
    add_check(
        "models.present",
        any(item["ok"] for item in model_checks),
        ", ".join(f"{item['horizon']}={'ok' if item['ok'] else 'missing'}" for item in model_checks) or "no horizons",
    )

    try:
        resp = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        add_check("binance.ping", resp.ok, f"http={resp.status_code}")
    except Exception as exc:
        add_check("binance.ping", False, str(exc))

    all_ok = all(item["ok"] for item in checks)
    return {
        "ok": all_ok,
        "asset": asset_u,
        "checks": checks,
        "models": model_checks,
        "meta": _build_run_meta(config=config, root=paths.root, dataset_hash=_hash_payload(model_checks)),
    }


def run_all(config: dict[str, Any], asset: str, root: str | Path = ".") -> dict[str, Any]:
    data_meta = run_data_pipeline(config, asset=asset, root=root)
    feats_meta = run_feature_label_pipeline(config, asset=asset, root=root)
    train_meta = run_training_pipeline(config, asset=asset, root=root)
    walk_meta = run_walk_forward_pipeline(config, asset=asset, root=root)
    sim_meta = run_simulation_pipeline(config, asset=asset, root=root)
    return {
        "data": data_meta,
        "features_labels": feats_meta,
        "training": train_meta,
        "walk_forward": walk_meta,
        "simulation": sim_meta,
    }
