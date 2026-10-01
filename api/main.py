"""
main.py  – FastAPI application entry point
Run with:  uvicorn api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from api.schemas import ForecastRequest, ForecastResponse, ForecastPoint, HealthResponse
from api.forecast_service import generate_forecasts, is_model_loaded, MODEL_VERSION

app = FastAPI(
    title="Store-Level Demand Forecaster API",
    description="Retail demand forecasting powered by Global LightGBM trained on M5 Walmart data.",
    version="1.0.0"
)

# ── Health endpoint ────────────────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health():
    loaded = is_model_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION
    )

# ── Model info ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/model/info", tags=["Model"])
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "algorithm": "Global LightGBM Regressor",
        "dataset": "M5 Forecasting Accuracy (Walmart)",
        "supported_horizon_days": "1–28",
        "features": ["lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28",
                     "sell_price", "wday", "month", "year", "snap flags", "identifiers"]
    }

# ── Forecast endpoint ──────────────────────────────────────────────────────────
@app.post("/api/v1/forecasts", response_model=ForecastResponse, tags=["Forecasts"])
def get_forecast(req: ForecastRequest):
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
    try:
        forecasts = generate_forecasts(req.store_id, req.item_id, req.horizon_days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return ForecastResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        model_version=MODEL_VERSION,
        forecasts=[ForecastPoint(**f) for f in forecasts]
    )
