from __future__ import annotations

from pathlib import Path
import unittest


class UiContractTests(unittest.TestCase):
    def test_ui_assets_use_cache_busting_query_suffixes(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        self.assertIn('/styles.css?v=lab_sweep_builder_v2', html)
        self.assertIn('/app.js?v=lab_sweep_builder_v2', html)

    def test_dashboard_uses_price_range_language(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8").lower()
        self.assertIn("price range", html)
        self.assertIn("median price", html)
        self.assertNotIn("quantiles", html)

    def test_sweep_builder_uses_structured_axis_editor(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        self.assertIn('id="builderSweepAxesList"', html)
        self.assertIn('id="builderSweepPreviewBody"', html)
        self.assertIn('id="builderAddSweepAxisBtn"', html)
        self.assertNotIn('id="builderSweepCoverages"', html)
        self.assertNotIn('id="builderSweepWindowDays"', html)
        self.assertNotIn('id="builderSweepFeatureSets"', html)
        self.assertNotIn('id="builderSweepCalibrationMethods"', html)
        self.assertNotIn('id="builderSweepHyperparams"', html)


if __name__ == "__main__":
    unittest.main()

