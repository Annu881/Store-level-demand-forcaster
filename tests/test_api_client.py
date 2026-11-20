"""Unit tests for the ForecasterApiClient in api_client.py."""

import unittest
from unittest.mock import patch, MagicMock
import httpx
from api_client import ForecasterApiClient


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ForecasterApiClient(base_url="http://127.0.0.1:8999", timeout=1.0)

    def test_offline_health_check_returns_false(self):
        # Port 8999 is closed, should cleanly return False without raising
        self.assertFalse(self.client.is_healthy(timeout=0.2))

    def test_custom_initialization(self):
        c = ForecasterApiClient("http://example.com/api/", timeout=5.5)
        self.assertEqual(c.base_url, "http://example.com/api")
        self.assertEqual(c.timeout, 5.5)

    @patch("httpx.Client.get")
    def test_mocked_health_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "service": "store-level-demand-forecaster"}
        mock_get.return_value = mock_resp

        self.assertTrue(self.client.is_healthy())
        health = self.client.get_health()
        self.assertEqual(health["status"], "ok")

    @patch("httpx.Client.get")
    def test_mocked_model_info(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"model_name": "direct_xgboost", "algorithm": "XGBoost"}
        mock_get.return_value = mock_resp

        info = self.client.get_model_info()
        self.assertEqual(info["model_name"], "direct_xgboost")

    @patch("httpx.Client.post")
    def test_mocked_replenishment_simulation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "run_id": "test-run-123",
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "result": {"suggested_order_quantity": 10},
        }
        mock_post.return_value = mock_resp

        payload = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "lead_time_days": 7,
            "on_hand": 5,
        }
        resp = self.client.simulate_replenishment(payload)
        self.assertEqual(resp["run_id"], "test-run-123")
        self.assertEqual(resp["result"]["suggested_order_quantity"], 10)


if __name__ == "__main__":
    unittest.main()
