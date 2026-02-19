"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a mapping at the root")
    return data


def load_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load and merge project configuration files."""
    base_dir = Path(config_dir)
    base = _read_yaml(base_dir / "base.yaml")
    assets = _read_yaml(base_dir / "assets.yaml")
    horizons = _read_yaml(base_dir / "horizons.yaml")
    feature_path = base_dir / "feature_sets" / f"{base.get('feature_set', 'v1')}.yaml"
    feature_set = _read_yaml(feature_path)

    merged = dict(base)
    merged["assets"] = assets.get("assets", {})
    merged["horizons_map"] = horizons.get("horizons", {})
    merged["feature_set_config"] = feature_set
    merged["config_dir"] = str(base_dir)
    return merged


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    data_root: Path
    artifacts_root: Path


def get_runtime_paths(config: dict[str, Any], root: str | Path = ".") -> RuntimePaths:
    root_path = Path(root).resolve()
    data_root = root_path / config.get("storage", {}).get("root", "data")
    artifacts_root = root_path / config.get("storage", {}).get("artifacts_root", "artifacts")
    data_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(root=root_path, data_root=data_root, artifacts_root=artifacts_root)

