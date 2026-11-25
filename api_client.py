"""HTTP Client for communicating with the FastAPI backend."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal
import httpx


logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv(
    "FORECASTER_API_URL",
    "https://store-level-demand-forcaster.onrender.com",
)
# Render free tier can take 30-60s to wake from sleep — use a generous timeout
DEFAULT_TIMEOUT = float(os.getenv("FORECASTER_API_TIMEOUT", "60.0"))


class ForecasterApiClient:
    """Client for FastAPI demand forecasting and replenishment endpoints."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def is_healthy(self, timeout: float = 15.0) -> bool:
        """Check if FastAPI backend is responding.
        Default 15 s — generous enough for Render free-tier cold-start."""
        try:
            with httpx.Client(base_url=self.base_url, timeout=timeout) as client:
                response = client.get("/health")
                return response.status_code == 200 and response.json().get("status") == "ok"
        except (httpx.RequestError, Exception):
            return False

    def get_health(self) -> dict[str, Any]:
        with self._client() as client:
            resp = client.get("/health")
            resp.raise_for_status()
            return resp.json()

    def get_model_info(self) -> dict[str, Any]:
        with self._client() as client:
            resp = client.get("/model")
            resp.raise_for_status()
            return resp.json()

    def get_evaluation_metrics(self) -> list[dict[str, Any]]:
        with self._client() as client:
            resp = client.get("/metrics")
            resp.raise_for_status()
            return resp.json()

    def get_data_quality(self) -> dict[str, Any]:
        with self._client() as client:
            resp = client.get("/data-quality")
            resp.raise_for_status()
            return resp.json()

    def get_forecasts(
        self,
        forecast_type: Literal["historical", "recursive", "direct"] = "recursive",
        store_id: str | None = None,
        item_id: str | None = None,
        horizon: int = 28,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "forecast_type": forecast_type,
            "horizon": horizon,
            "limit": limit,
        }
        if store_id:
            params["store_id"] = store_id
        if item_id:
            params["item_id"] = item_id

        with self._client() as client:
            resp = client.get("/forecasts", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_segments(self, by: Literal["category", "store_id"] = "category") -> list[dict[str, Any]]:
        with self._client() as client:
            resp = client.get("/analytics/segments", params={"by": by})
            resp.raise_for_status()
            return resp.json()

    def get_residuals(
        self,
        store_id: str | None = None,
        item_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if store_id:
            params["store_id"] = store_id
        if item_id:
            params["item_id"] = item_id

        with self._client() as client:
            resp = client.get("/residuals", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_feature_importance(self) -> list[dict[str, Any]]:
        with self._client() as client:
            resp = client.get("/feature-importance")
            resp.raise_for_status()
            return resp.json()

    def simulate_replenishment(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._client() as client:
            resp = client.post("/replenishment", json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_replenishment_run(self, run_id: str) -> dict[str, Any]:
        with self._client() as client:
            resp = client.get(f"/replenishment/{run_id}")
            resp.raise_for_status()
            return resp.json()


# Module-level default instance
default_client = ForecasterApiClient()
