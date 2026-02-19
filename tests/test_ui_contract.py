from __future__ import annotations

from pathlib import Path
import unittest


class UiContractTests(unittest.TestCase):
    def test_dashboard_uses_price_range_language(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8").lower()
        self.assertIn("price range", html)
        self.assertIn("median price", html)
        self.assertNotIn("quantiles", html)


if __name__ == "__main__":
    unittest.main()

