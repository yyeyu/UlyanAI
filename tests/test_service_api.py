from __future__ import annotations

import unittest

from src.service.api import get_health, get_predict, post_predict_batch
from src.service.schemas import PredictBatchRequest, PredictBatchRequestItem


class ServiceContractTests(unittest.TestCase):
    def test_health_contract_contains_fallback_flags(self) -> None:
        payload = get_health().model_dump()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "price_ranges")
        self.assertIn("fallback_enabled", payload)
        self.assertIn("fallback_in_use", payload)
        self.assertIn("missing_models", payload)

    def test_predict_uses_price_ranges_only(self) -> None:
        response = get_predict(asset="BTC", horizon="1h")
        payload = response.model_dump()
        self.assertEqual(payload["mode"], "price_ranges")
        self.assertIn("price_levels", payload)
        self.assertIn("price_range", payload)
        self.assertIn("median_price", payload)
        self.assertNotIn("quantiles", payload)
        self.assertLessEqual(payload["price_levels"]["p10"], payload["price_levels"]["p50"])
        self.assertLessEqual(payload["price_levels"]["p50"], payload["price_levels"]["p90"])

    def test_predict_batch_contract(self) -> None:
        batch_request = PredictBatchRequest(
            requests=[
                PredictBatchRequestItem(asset="BTC", horizon="5m"),
                PredictBatchRequestItem(asset="ETH", horizon="1d"),
            ]
        )
        response = post_predict_batch(batch_request)
        payload = response.model_dump()
        self.assertEqual(len(payload["predictions"]), 2)
        for item in payload["predictions"]:
            self.assertEqual(item["mode"], "price_ranges")
            self.assertIn("price_range", item)
            self.assertNotIn("quantiles", item)


if __name__ == "__main__":
    unittest.main()
