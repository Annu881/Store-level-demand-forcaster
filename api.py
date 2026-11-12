"""FastAPI interface for saved M5 forecasts, evaluation, and simulations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Generator, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from inventory import SERVICE_Z, ReplenishmentInputs, simulate_replenishment


ROOT = Path(__file__).resolve().parent
MODEL_READY = ROOT / "data" / "model_ready"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
DATABASE_FILE = MODEL_READY / "forecast_app.sqlite"
DATA_QUALITY_FILE = ROOT / "data" / "quality" / "latest.json"
MODEL_METADATA_FILE = ROOT / "models" / "direct_xgboost_metadata.json"
FORECAST_PATHS = {
    "historical": MODEL_READY / "final_forecast_d1914_d1941.parquet",
    "recursive": MODEL_READY / "future_forecast_d1942_d1969.parquet",
    "direct": MODEL_READY / "direct_forecast_d1942_d1969.parquet",
}

app = FastAPI(
    title="Store-Level Demand Forecaster",
    description=(
        "Browse saved M5 demand forecasts and evaluation results. "
        "Replenishment output is a simulation and requires human review."
    ),
    version="1.0.0",
)


class ReplenishmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_id: str = Field(pattern=r"^(CA|TX|WI)_[0-9]{1,2}$")
    item_id: str = Field(pattern=r"^(FOODS|HOBBIES|HOUSEHOLD)_[0-9]+_[0-9]+$")
    forecast_type: Literal["recursive", "direct"] = "recursive"
    lead_time_days: int = Field(ge=1, le=28)
    review_period_days: int = Field(default=0, ge=0, le=27)
    on_hand: float = Field(ge=0, allow_inf_nan=False)
    on_order: float = Field(default=0, ge=0, allow_inf_nan=False)
    backorders: float = Field(default=0, ge=0, allow_inf_nan=False)
    pack_size: int = Field(default=1, ge=1)
    minimum_order_quantity: int = Field(default=0, ge=0)
    service_level: float = Field(default=0.95, ge=0.8, le=0.99)
    daily_residual_std: float = Field(default=0, ge=0, allow_inf_nan=False)

    @field_validator("service_level")
    @classmethod
    def validate_service_level(cls, value: float) -> float:
        if value not in SERVICE_Z:
            raise ValueError(f"service_level must be one of {sorted(SERVICE_Z)}.")
        return value


@lru_cache(maxsize=4)
def _read_forecast(kind: str) -> pd.DataFrame:
    if kind not in FORECAST_PATHS:
        raise ValueError(f"Unknown forecast type: {kind}")
    path = FORECAST_PATHS[kind]
    if not path.is_file():
        raise FileNotFoundError(f"{kind.title()} forecast is not available: {path}")
    frame = pd.read_parquet(path)
    if kind == "direct":
        required = {
            "id",
            "store_id",
            "item_id",
            "forecast_date_day",
            "forecast_date",
            "horizon",
            "point_forecast",
            "model_version",
            "data_version",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Direct forecast is missing columns: {sorted(missing)}")
        frame = frame.rename(
            columns={
                "forecast_date_day": "d",
                "forecast_date": "date",
                "point_forecast": "predicted_sales",
            }
        )
        frame["horizon"] = frame["horizon"].astype(int)
    else:
        required = {"id", "d", "predicted_sales"}
        if kind == "historical":
            required.add("actual_sales")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        day_numbers = frame["d"].str.extract(r"^d_([0-9]+)$")[0]
        if day_numbers.isna().any():
            raise ValueError(f"{path.name} contains an invalid day label.")
        frame["horizon"] = (
            day_numbers.astype(int) - day_numbers.astype(int).min() + 1
        )
        parsed = frame["id"].str.extract(
            r"^(?P<category>[^_]+)_(?P<department>[^_]+)_(?P<item_number>[^_]+)_"
            r"(?P<state>CA|TX|WI)_(?P<store_number>[0-9]+)_evaluation$"
        )
        if parsed.isna().to_numpy().any():
            raise ValueError("Forecast contains an invalid M5 series ID.")
        frame["category"] = parsed["category"]
        frame["store_id"] = parsed["state"] + "_" + parsed["store_number"]
        frame["item_id"] = (
            parsed["category"] + "_" + parsed["department"] + "_" + parsed["item_number"]
        )
        if "date" not in frame.columns:
            if not CALENDAR_FILE.is_file():
                raise FileNotFoundError(f"Calendar data not found: {CALENDAR_FILE}")
            calendar = pd.read_csv(CALENDAR_FILE, usecols=["d", "date"])
            frame = frame.merge(calendar, on="d", how="left", validate="many_to_one")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["predicted_sales"].isna().any() or (frame["predicted_sales"] < 0).any():
        raise ValueError("Forecast predictions must be present and non-negative.")
    return frame


def _get_forecast(kind: str) -> pd.DataFrame:
    try:
        return _read_forecast(kind)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _database() -> sqlite3.Connection:
    MODEL_READY.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS replenishment_runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            store_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            forecast_type TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return connection


@contextmanager
def _open_database() -> Generator[sqlite3.Connection, None, None]:
    connection = _database()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "store-level-demand-forecaster"}


@app.get("/model")
def model_info() -> dict[str, object]:
    if MODEL_METADATA_FILE.is_file():
        return json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    return {
        "model_name": "recursive_xgboost",
        "status": "legacy saved forecasts are available; direct model is not trained",
        "direct_model_trained": False,
    }


@app.get("/metrics")
def evaluation_metrics() -> list[dict[str, object]]:
    path = MODEL_READY / "rolling_backtest_metrics.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Rolling evaluation metrics are unavailable.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/data-quality")
def data_quality() -> dict[str, object]:
    if not DATA_QUALITY_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="No data-quality report exists. Run data_quality.py first.",
        )
    return json.loads(DATA_QUALITY_FILE.read_text(encoding="utf-8"))


@app.get("/forecasts")
def forecasts(
    forecast_type: Literal["historical", "recursive", "direct"] = "recursive",
    store_id: str | None = None,
    item_id: str | None = None,
    horizon: int | None = Query(default=None, ge=1, le=28),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, object]:
    frame = _get_forecast(forecast_type)
    if store_id is not None:
        frame = frame.loc[frame["store_id"] == store_id]
    if item_id is not None:
        frame = frame.loc[frame["item_id"] == item_id]
    if horizon is not None:
        if "horizon" not in frame:
            raise HTTPException(status_code=400, detail="Horizon filtering is unavailable.")
        frame = frame.loc[frame["horizon"] == horizon]
    rows = frame.head(limit).copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    return {
        "forecast_type": forecast_type,
        "row_count": len(rows),
        "total_matching_rows": len(frame),
        "rows": rows.to_dict(orient="records"),
    }


@app.get("/analytics/segments")
def segment_metrics(
    by: Literal["category", "store_id"] = "category",
) -> list[dict[str, object]]:
    frame = _get_forecast("historical")
    if "category" not in frame:
        parsed = frame["id"].str.extract(r"^(?P<category>[^_]+)_")
        frame["category"] = parsed["category"]
    frame["error"] = frame["predicted_sales"] - frame["actual_sales"]
    results = []
    for key, group in frame.groupby(by, sort=True):
        actual_total = float(group["actual_sales"].abs().sum())
        results.append({
            by: key,
            "MAE": float(group["error"].abs().mean()),
            "RMSE": float((group["error"].pow(2).mean()) ** 0.5),
            "WAPE_pct": (
                float(100 * group["error"].abs().sum() / actual_total)
                if actual_total
                else None
            ),
            "Bias": float(group["error"].mean()),
            "row_count": int(len(group)),
        })
    return results


@app.get("/residuals")
def residuals(
    store_id: str | None = None,
    item_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, object]:
    frame = _get_forecast("historical")
    if store_id is not None:
        frame = frame.loc[frame["store_id"] == store_id]
    if item_id is not None:
        frame = frame.loc[frame["item_id"] == item_id]
    frame = frame.head(limit).copy()
    frame["residual"] = frame["predicted_sales"] - frame["actual_sales"]
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return {"row_count": len(frame), "rows": frame.to_dict(orient="records")}


@app.get("/feature-importance")
def feature_importance() -> list[dict[str, object]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="XGBoost is not installed.") from exc
    model_path = ROOT / "models" / "direct_xgboost.json"
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Train the direct XGBoost model to produce feature importance.",
        )
    model = xgb.Booster()
    model.load_model(str(model_path))
    scores = model.get_score(importance_type="gain")
    return [
        {"feature": feature, "gain": float(gain)}
        for feature, gain in sorted(scores.items(), key=lambda row: row[1], reverse=True)
    ]


@app.post("/replenishment", status_code=201)
def replenishment_simulation(request: ReplenishmentRequest) -> dict[str, object]:
    frame = _get_forecast(request.forecast_type)
    selected = frame.loc[
        (frame["store_id"] == request.store_id) & (frame["item_id"] == request.item_id)
    ].sort_values("date")
    if selected.empty:
        raise HTTPException(status_code=404, detail="No forecast found for this item and store.")
    protection_days = request.lead_time_days + request.review_period_days
    if protection_days > 28:
        raise HTTPException(
            status_code=422,
            detail="Lead time plus review period cannot exceed the 28-day forecast horizon.",
        )
    forecasts = selected["predicted_sales"].astype(float).tolist()[:protection_days]
    if len(forecasts) < protection_days:
        raise HTTPException(
            status_code=422,
            detail="Selected forecast does not cover the configured protection period.",
        )
    inputs = ReplenishmentInputs(
        on_hand=request.on_hand,
        on_order=request.on_order,
        backorders=request.backorders,
        lead_time_days=request.lead_time_days,
        review_period_days=request.review_period_days,
        pack_size=request.pack_size,
        minimum_order_quantity=request.minimum_order_quantity,
        service_level=request.service_level,
        daily_residual_std=request.daily_residual_std,
    )
    result = simulate_replenishment(forecasts, inputs)
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _open_database() as connection:
        connection.execute(
            """
            INSERT INTO replenishment_runs (
                run_id, created_at_utc, store_id, item_id, forecast_type,
                inputs_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                request.store_id,
                request.item_id,
                request.forecast_type,
                request.model_dump_json(),
                json.dumps(result),
            ),
        )
    return {
        "run_id": run_id,
        "created_at_utc": created_at,
        "forecast_type": request.forecast_type,
        "store_id": request.store_id,
        "item_id": request.item_id,
        "result": result,
    }


@app.get("/replenishment/{run_id}")
def get_replenishment_run(run_id: str) -> dict[str, object]:
    with _open_database() as connection:
        row = connection.execute(
            "SELECT * FROM replenishment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Replenishment simulation was not found.")
    return {
        "run_id": row["run_id"],
        "created_at_utc": row["created_at_utc"],
        "store_id": row["store_id"],
        "item_id": row["item_id"],
        "forecast_type": row["forecast_type"],
        "inputs": json.loads(row["inputs_json"]),
        "result": json.loads(row["result_json"]),
    }
