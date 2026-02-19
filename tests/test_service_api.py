from __future__ import annotations

import unittest

from src.service.api import get_predict, post_predict_batch
from src.service.schemas import PredictBatchRequest, PredictBatchRequestItem


class ServiceContractTests(unittest.TestCase):
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
