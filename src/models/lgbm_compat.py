"""Compatibility helpers for LightGBM text model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_FEATURE_INFOS_SAFE_LINE_LEN = 1024


def _replace_feature_infos_if_needed(model_text: str) -> str:
    """Replace overlong feature_infos rows with `none` tokens.

    Some LightGBM builds on Windows fail to parse text models when the
    `feature_infos=` line is too long. Replacing this line with neutral
    placeholders keeps inference behavior while avoiding parser failures.
    """
    lines = model_text.splitlines()
    max_feature_idx: int | None = None
    feature_infos_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.startswith("max_feature_idx="):
            try:
                max_feature_idx = int(line.split("=", 1)[1])
            except ValueError:
                max_feature_idx = None
        elif line.startswith("feature_infos="):
            feature_infos_idx = idx
            break

    if feature_infos_idx is None or max_feature_idx is None:
        return model_text

    if len(lines[feature_infos_idx]) <= _FEATURE_INFOS_SAFE_LINE_LEN:
        return model_text

    feature_count = max_feature_idx + 1
    lines[feature_infos_idx] = "feature_infos=" + " ".join(["none"] * feature_count)
    normalized = "\n".join(lines)
    if model_text.endswith(("\n", "\r\n")):
        normalized += "\n"
    return normalized


def sanitize_model_file_in_place(model_path: Path) -> bool:
    """Normalize model text in-place if it has incompatible feature_infos."""
    text = model_path.read_text(encoding="utf-8")
    normalized = _replace_feature_infos_if_needed(text)
    if normalized == text:
        return False
    model_path.write_text(normalized, encoding="utf-8", newline="\n")
    return True


def load_booster_compat(lgb_module: Any, model_path: Path) -> Any:
    """Load LightGBM model with compatibility normalization when needed."""
    text = model_path.read_text(encoding="utf-8")
    normalized = _replace_feature_infos_if_needed(text)
    if normalized != text:
        return lgb_module.Booster(model_str=normalized)
    return lgb_module.Booster(model_file=str(model_path))
