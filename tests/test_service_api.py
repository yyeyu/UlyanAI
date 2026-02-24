from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import src.service.api as service_api
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

    def test_predict_prefers_live_spot_quote_for_price(self) -> None:
        feature_ts = pd.Timestamp("2026-01-01T00:00:00+00:00")
        live_ts = pd.Timestamp("2026-01-01T00:00:05+00:00")
        with (
            patch.object(service_api, "_load_bundle_cached", return_value=None),
            patch.object(
                service_api.CANDLE_CACHE,
                "latest_feature_row",
                return_value=({}, 100.0, feature_ts, False),
            ),
            patch.object(
                service_api.CANDLE_CACHE,
                "latest_spot_quote",
                return_value=(101.25, live_ts, False),
            ),
        ):
            response = service_api.get_predict(asset="BTC", horizon="1h")
        self.assertAlmostEqual(response.price_spot, 101.25, places=8)
        self.assertTrue(response.as_of.startswith("2026-01-01T00:00:05"))


if __name__ == "__main__":
    unittest.main()
