"""Model artifact registry helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.utils import dump_json, load_json


def build_model_version(asset: str, horizon: str, model: str, mode: str, feature_set: str) -> str:
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{asset.lower()}_{horizon}_{model}_{mode}_{feature_set}_{date_tag}"


def model_dir(artifacts_root: Path, asset: str, horizon: str, model_version: str) -> Path:
    return artifacts_root / "models" / asset.upper() / horizon / model_version


def write_metadata(path: Path, metadata: dict) -> None:
    dump_json(path / "metadata.json", metadata)


def read_metadata(path: Path) -> dict:
    return load_json(path / "metadata.json")


def latest_model_dir(artifacts_root: Path, asset: str, horizon: str) -> Path | None:
    base = artifacts_root / "models" / asset.upper() / horizon
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        return None
    return candidates[-1]

