"""
tests/test_api.py
Unit tests for the FastAPI Demand Forecaster endpoints.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_model_info_endpoint():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "Global LightGBM Regressor"
    assert "features" in data
    assert len(data["features"]) > 0

def test_forecast_endpoint_invalid_store():
    # Sending missing fields should trigger a validation error
    response = client.post(
        "/api/v1/forecasts",
        json={"store_id": "CA_1"} # missing item_id
    )
    assert response.status_code == 422 # Pydantic Validation Error

def test_forecast_horizon_validation():
    # Horizon is strictly capped between 1 and 28
    response = client.post(
        "/api/v1/forecasts",
        json={"store_id": "CA_1", "item_id": "HOBBIES_1_001", "horizon_days": 100}
    )
    assert response.status_code == 422 # Pydantic Validation constraint failure
