"""Command-line interface for project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn

from src.config import load_config
from src.pipeline import (
    run_all,
    run_data_pipeline,
    run_feature_label_pipeline,
    run_simulation_pipeline,
    run_training_pipeline,
    run_walk_forward_pipeline,
)


def _load_cfg(path: str) -> dict[str, Any]:
    return load_config(path)


def cmd_data_sync(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_data_pipeline(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_features_build(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_feature_label_pipeline(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_train_run(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_training_pipeline(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_eval_walk_forward(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_walk_forward_pipeline(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_sim_run(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_simulation_pipeline(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_pipeline_all(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    result = run_all(cfg, asset=args.asset, root=args.root)
    print(result)


def cmd_service_run(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config_dir)
    uvicorn.run(
        "src.service.api:app",
        host=str(cfg.get("service", {}).get("host", "0.0.0.0")),
        port=int(cfg.get("service", {}).get("port", 8000)),
        reload=bool(args.reload),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ulyanai")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--root", default=".")
    parser.add_argument("--asset", default="BTC")

    sub = parser.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data-sync", help="Run data download + clean + resample + QC")
    data.set_defaults(func=cmd_data_sync)

    feats = sub.add_parser("features-build", help="Build features and labels")
    feats.set_defaults(func=cmd_features_build)

    train = sub.add_parser("train-run", help="Train quantile models")
    train.set_defaults(func=cmd_train_run)

    eval_cmd = sub.add_parser("eval-walk-forward", help="Run walk-forward evaluation")
    eval_cmd.set_defaults(func=cmd_eval_walk_forward)

    sim = sub.add_parser("sim-run", help="Run paper simulation")
    sim.set_defaults(func=cmd_sim_run)

    pipeline = sub.add_parser("pipeline-all", help="Run full pipeline")
    pipeline.set_defaults(func=cmd_pipeline_all)

    service = sub.add_parser("service-run", help="Start FastAPI service")
    service.add_argument("--reload", action="store_true")
    service.set_defaults(func=cmd_service_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

