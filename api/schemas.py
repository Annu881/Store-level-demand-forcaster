from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class ForecastRequest(BaseModel):
    store_id: str = Field(..., example="CA_1")
    item_id: str = Field(..., example="HOBBIES_1_001")
    horizon_days: int = Field(default=28, ge=1, le=28, example=28)

class ForecastPoint(BaseModel):
    forecast_date: date
    horizon: int
    point_forecast: float

class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    model_version: str
    forecasts: List[ForecastPoint]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
