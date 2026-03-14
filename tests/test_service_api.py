from __future__ import annotations

from datetime import datetime, timezone
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
            patch.object(service_api, "_production_model_id", return_value=None),
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

    def test_predict_without_model_id_prefers_registry_production_model(self) -> None:
        feature_ts = pd.Timestamp("2026-01-01T00:00:00+00:00")

        class DummyBundle:
            metadata = {"calibrator": {"target_coverage": 0.8}}

        class DummyPrediction:
            def __init__(self) -> None:
                self.primary = service_api.ReturnQuantiles(q10=-0.01, q50=0.0, q90=0.01)
                self.extra_quantiles = {}

        with (
            patch.object(service_api, "_production_model_id", return_value="prod-btc-5m"),
            patch.object(service_api, "_load_bundle_by_id_cached", return_value=DummyBundle()),
            patch.object(service_api, "_load_bundle_cached") as load_latest_mock,
            patch.object(
                service_api.CANDLE_CACHE,
                "latest_feature_row",
                return_value=({"close": 100.0}, 100.0, feature_ts, False),
            ),
            patch.object(
                service_api.CANDLE_CACHE,
                "latest_spot_quote",
                side_effect=RuntimeError("no live quote"),
            ),
            patch.object(
                service_api,
                "predict_bundle_outputs",
                return_value=DummyPrediction(),
            ),
        ):
            response = service_api.get_predict(asset="BTC", horizon="5m")
        self.assertEqual(response.model_version, "prod-btc-5m")
        load_latest_mock.assert_not_called()

    def test_polymarket_crypto_events_endpoint_paginates_and_normalizes(self) -> None:
        class DummyResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        calls: list[dict[str, object]] = []
        page0 = [
            {
                "id": "101",
                "slug": "btc-over-100k",
                "title": "BTC over 100k this year?",
                "active": True,
                "closed": False,
                "startDate": "2026-01-01T00:00:00Z",
                "endDate": "2026-12-31T00:00:00Z",
                "volume": "1234.56",
                "liquidityClob": "789.01",
                "openInterest": "456.78",
                "tags": [{"label": "Crypto", "slug": "crypto"}],
                "markets": [
                    {
                        "id": "m-1",
                        "conditionId": "cond-1",
                        "question": "Will BTC close above 100k?",
                        "slug": "btc-close-above-100k",
                        "active": True,
                        "closed": False,
                        "outcomes": "[\"Yes\", \"No\"]",
                        "outcomePrices": "[\"0.42\", \"0.58\"]",
                        "clobTokenIds": "[\"yes-token\", \"no-token\"]",
                        "volumeNum": "777.7",
                    }
                ],
            }
        ]
        page1 = [
            {
                "id": "102",
                "slug": "eth-over-10k",
                "title": "ETH over 10k this year?",
                "active": True,
                "closed": False,
                "markets": [],
            }
        ]
        page2 = []

        def fake_get(url, params=None, timeout=None):
            self.assertIn("/events", url)
            calls.append(dict(params or {}))
            offset = int((params or {}).get("offset", 0))
            if offset == 0:
                return DummyResponse(page0)
            if offset == 1:
                return DummyResponse(page1)
            return DummyResponse(page2)

        with (
            patch.object(service_api, "POLYMARKET_EVENTS_PAGE_SIZE", 1),
            patch.object(service_api, "POLYMARKET_EVENTS_MAX_PAGES", 10),
            patch.object(service_api.requests, "get", side_effect=fake_get),
        ):
            response = service_api.polymarket_crypto_events(active=True, closed=False)

        self.assertEqual(response.source, "polymarket_gamma")
        self.assertEqual(response.tag_slug, "crypto")
        self.assertEqual(response.total, 2)
        self.assertEqual([item.get("offset") for item in calls], [0, 1, 2])
        first = response.items[0]
        self.assertEqual(first.event_id, "101")
        self.assertEqual(first.tags, ["Crypto"])
        self.assertEqual(first.markets_count, 1)
        self.assertAlmostEqual(first.volume or 0.0, 1234.56, places=8)
        market = first.markets[0]
        self.assertEqual(market.yes_token_id, "yes-token")
        self.assertEqual(market.no_token_id, "no-token")
        self.assertEqual(market.outcomes, ["Yes", "No"])
        self.assertEqual(market.outcome_prices, [0.42, 0.58])

    def test_build_event_metrics_adds_interval_metrics_and_wis(self) -> None:
        event = {
            "pred_low": 90.0,
            "pred_mid": 100.0,
            "pred_high": 110.0,
            "price_t0": 100.0,
            "expires_at": "2026-01-01T00:00:00+00:00",
            "prediction": {
                "price_range": {"nominal": 0.80},
                "quantiles_pred": {
                    "q10": 90.0,
                    "q25": 95.0,
                    "q50": 100.0,
                    "q75": 105.0,
                    "q90": 110.0,
                },
            },
        }
        metrics = service_api._build_event_metrics(
            event,
            actual_price=108.0,
            now_utc=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        )

        self.assertEqual(metrics["miss_side"], "inside")
        self.assertAlmostEqual(metrics["center_error"], 8.0, places=8)
        self.assertAlmostEqual(metrics["interval_score"], 20.0, places=8)
        self.assertAlmostEqual(metrics["wis"], 4.6, places=8)
        self.assertEqual(metrics["latency_ms"], 2000)

    def test_build_event_metrics_marks_miss_side_above_and_below(self) -> None:
        event = {
            "pred_low": 95.0,
            "pred_mid": 100.0,
            "pred_high": 105.0,
            "price_t0": 100.0,
            "expires_at": "2026-01-01T00:00:00+00:00",
            "prediction": {"price_range": {"nominal": 0.80}, "quantiles_pred": {}},
        }

        below = service_api._build_event_metrics(
            event,
            actual_price=90.0,
            now_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        above = service_api._build_event_metrics(
            event,
            actual_price=110.0,
            now_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(below["miss_side"], "below")
        self.assertEqual(above["miss_side"], "above")
        self.assertGreater(below["interval_score"], 0.0)
        self.assertGreater(above["interval_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
