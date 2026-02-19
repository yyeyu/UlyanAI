from __future__ import annotations

import math
import unittest

from src.models.predict import (
    NonMonotonicPriceError,
    ReturnQuantiles,
    build_price_prediction,
    returns_to_price_levels,
)


class PredictConversionTests(unittest.TestCase):
    def test_return_to_price_formula(self) -> None:
        levels = returns_to_price_levels(
            price_spot=100.0,
            q10_return=math.log(0.9),
            q50_return=math.log(1.0),
            q90_return=math.log(1.1),
        )
        self.assertAlmostEqual(levels.p10, 90.0, places=8)
        self.assertAlmostEqual(levels.p50, 100.0, places=8)
        self.assertAlmostEqual(levels.p90, 110.0, places=8)

    def test_non_monotonic_levels_raise(self) -> None:
        with self.assertRaises(NonMonotonicPriceError):
            returns_to_price_levels(
                price_spot=100.0,
                q10_return=math.log(1.1),
                q50_return=math.log(1.0),
                q90_return=math.log(0.9),
            )

    def test_price_range_low_high(self) -> None:
        payload = build_price_prediction(
            asset="BTC",
            horizon="1h",
            as_of=None,
            price_spot=50000.0,
            return_quantiles=ReturnQuantiles(q10=-0.01, q50=0.0, q90=0.02),
            nominal=0.80,
            model_version="btc_1h_lgbm_q_v1_2026-02-18",
        )
        self.assertEqual(payload["mode"], "price_ranges")
        self.assertNotIn("quantiles", payload)
        self.assertLessEqual(payload["price_range"]["low"], payload["price_range"]["high"])
        self.assertEqual(payload["price_range"]["low"], payload["price_levels"]["p10"])
        self.assertEqual(payload["price_range"]["high"], payload["price_levels"]["p90"])


if __name__ == "__main__":
    unittest.main()

