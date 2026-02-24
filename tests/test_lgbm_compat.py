from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.lgbm_compat import load_booster_compat, sanitize_model_file_in_place


def _build_model_text(*, max_feature_idx: int, feature_infos_line: str) -> str:
    return (
        "tree\n"
        "version=v4\n"
        "num_class=1\n"
        f"max_feature_idx={max_feature_idx}\n"
        f"feature_infos={feature_infos_line}\n"
        "tree_sizes=1\n"
    )


class _FakeBooster:
    def __init__(self, *, model_file: str | None = None, model_str: str | None = None) -> None:
        self.model_file = model_file
        self.model_str = model_str


class _FakeLGB:
    Booster = _FakeBooster


class LightGBMCompatTests(unittest.TestCase):
    def test_sanitize_model_file_in_place_rewrites_long_feature_infos(self) -> None:
        text = _build_model_text(max_feature_idx=2, feature_infos_line="x" * 1100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            path.write_text(text, encoding="utf-8")
            changed = sanitize_model_file_in_place(path)
            self.assertTrue(changed)
            normalized = path.read_text(encoding="utf-8")
            self.assertIn("feature_infos=none none none", normalized)

    def test_load_booster_compat_uses_model_str_for_long_feature_infos(self) -> None:
        text = _build_model_text(max_feature_idx=1, feature_infos_line="x" * 1100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            path.write_text(text, encoding="utf-8")
            booster = load_booster_compat(_FakeLGB, path)
            self.assertIsNone(booster.model_file)
            self.assertIsNotNone(booster.model_str)
            self.assertIn("feature_infos=none none", booster.model_str)

    def test_load_booster_compat_uses_model_file_for_short_feature_infos(self) -> None:
        text = _build_model_text(max_feature_idx=1, feature_infos_line="[0:1] [0:1]")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            path.write_text(text, encoding="utf-8")
            booster = load_booster_compat(_FakeLGB, path)
            self.assertEqual(booster.model_file, str(path))
            self.assertIsNone(booster.model_str)


if __name__ == "__main__":
    unittest.main()
