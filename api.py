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


def _parse_m5_metadata(frame: pd.DataFrame, kind: str, day_min: int | None = None) -> pd.DataFrame:
    """Validate and enrich legacy M5 forecast rows with horizon, store_id, item_id, and date."""
    required = {"id", "d", "predicted_sales"}
    if kind == "historical":
        required.add("actual_sales")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{FORECAST_PATHS[kind].name} is missing columns: {sorted(missing)}")
    if frame.empty:
        return frame.assign(
            horizon=pd.Series(dtype="int64"),
            category=pd.Series(dtype="object"),
            store_id=pd.Series(dtype="object"),
            item_id=pd.Series(dtype="object"),
            date=pd.to_datetime(pd.Series(dtype="object")),
        )
    frame = frame.copy()
    day_numbers = frame["d"].str.extract(r"^d_([0-9]+)$")[0]
    if day_numbers.isna().any():
        raise ValueError(f"{FORECAST_PATHS[kind].name} contains an invalid day label.")
    base_day = day_min if day_min is not None else int(day_numbers.astype(int).min())
    frame["horizon"] = day_numbers.astype(int) - base_day + 1
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
    return frame


def _read_forecast_filtered(
    kind: str,
    store_id: str | None = None,
    item_id: str | None = None,
    horizon: int | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Read forecast parquet with pushdown/pre-slice filtering to stay within 512 MB RAM."""
    if kind not in FORECAST_PATHS:
        raise ValueError(f"Unknown forecast type: {kind}")
    path = FORECAST_PATHS[kind]
    if not path.is_file():
        raise FileNotFoundError(f"{kind.title()} forecast is not available: {path}")

    if kind == "direct":
        filters: list = []
        if store_id is not None:
            filters.append(("store_id", "==", store_id))
        if item_id is not None:
            filters.append(("item_id", "==", item_id))
        if horizon is not None:
            filters.append(("horizon", "==", horizon))
        frame = pd.read_parquet(path, filters=filters if filters else None)
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
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame["predicted_sales"].isna().any() or (frame["predicted_sales"] < 0).any():
            raise ValueError("Forecast predictions must be present and non-negative.")
        total_matching = len(frame)
        if limit is not None:
            frame = frame.head(limit).copy()
        return frame, total_matching

    # Legacy parquets ("historical" and "recursive") only store `id` (not `store_id`/`item_id`)
    filters = []
    if store_id is not None and item_id is not None:
        filters.append(("id", "==", f"{item_id}_{store_id}_evaluation"))
    frame = pd.read_parquet(path, filters=filters if filters else None)

    if store_id is not None and item_id is None:
        frame = frame.loc[frame["id"].str.endswith(f"_{store_id}_evaluation")]
    elif item_id is not None and store_id is None:
        frame = frame.loc[frame["id"].str.startswith(f"{item_id}_")]

    day_min: int | None = None
    if not frame.empty:
        day_nums = frame["d"].str.slice(2).astype(int)
        day_min = int(day_nums.min())
        if horizon is not None:
            frame = frame.loc[day_nums == (day_min + horizon - 1)]

    total_matching = len(frame)
    if limit is not None:
        frame = frame.head(limit)

    frame = _parse_m5_metadata(frame, kind, day_min=day_min)
    if frame["predicted_sales"].isna().any() or (frame["predicted_sales"] < 0).any():
        raise ValueError("Forecast predictions must be present and non-negative.")
    return frame, total_matching


def _read_forecast(kind: str) -> pd.DataFrame:
    frame, _ = _read_forecast_filtered(kind)
    return frame


def _get_forecast_filtered(
    kind: str,
    store_id: str | None = None,
    item_id: str | None = None,
    horizon: int | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, int]:
    try:
        return _read_forecast_filtered(
            kind,
            store_id=store_id,
            item_id=item_id,
            horizon=horizon,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _get_forecast(kind: str) -> pd.DataFrame:
    frame, _ = _get_forecast_filtered(kind)
    return frame


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
    horizon_val = horizon if isinstance(horizon, int) else None
    limit_count = limit if isinstance(limit, int) else getattr(limit, "default", 100)
    rows, total_matching = _get_forecast_filtered(
        forecast_type,
        store_id=store_id,
        item_id=item_id,
        horizon=horizon_val,
        limit=limit_count,
    )
    rows = rows.copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    return {
        "forecast_type": forecast_type,
        "row_count": len(rows),
        "total_matching_rows": total_matching,
        "rows": rows.to_dict(orient="records"),
    }


@app.get("/analytics/segments")
def segment_metrics(
    by: Literal["category", "store_id"] = "category",
) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    path = FORECAST_PATHS["historical"]
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Historical forecast is not available: {path}")

    stats: dict[str, dict[str, float]] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=100_000, columns=["id", "predicted_sales", "actual_sales"]):
        chunk = batch.to_pandas()
        if by == "category":
            chunk[by] = chunk["id"].str.split("_", n=1).str[0]
        else:
            parts = chunk["id"].str.rsplit("_", n=3)
            chunk[by] = parts.str[-3] + "_" + parts.str[-2]
        chunk["error"] = chunk["predicted_sales"] - chunk["actual_sales"]
        chunk["abs_error"] = chunk["error"].abs()
        chunk["sq_error"] = chunk["error"].pow(2)
        chunk["abs_actual"] = chunk["actual_sales"].abs()
        grouped = chunk.groupby(by, sort=False).agg(
            abs_error=("abs_error", "sum"),
            sq_error=("sq_error", "sum"),
            error=("error", "sum"),
            abs_actual=("abs_actual", "sum"),
            count=("error", "count"),
        )
        for key, row in grouped.iterrows():
            acc = stats.setdefault(
                str(key),
                {"abs_error": 0.0, "sq_error": 0.0, "error": 0.0, "abs_actual": 0.0, "count": 0.0},
            )
            acc["abs_error"] += float(row["abs_error"])
            acc["sq_error"] += float(row["sq_error"])
            acc["error"] += float(row["error"])
            acc["abs_actual"] += float(row["abs_actual"])
            acc["count"] += float(row["count"])

    results = []
    for key in sorted(stats):
        acc = stats[key]
        n = acc["count"]
        actual_total = acc["abs_actual"]
        results.append({
            by: key,
            "MAE": float(acc["abs_error"] / n) if n else 0.0,
            "RMSE": float((acc["sq_error"] / n) ** 0.5) if n else 0.0,
            "WAPE_pct": float(100 * acc["abs_error"] / actual_total) if actual_total else None,
            "Bias": float(acc["error"] / n) if n else 0.0,
            "row_count": int(n),
        })
    return results


@app.get("/residuals")
def residuals(
    store_id: str | None = None,
    item_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, object]:
    limit_count = limit if isinstance(limit, int) else getattr(limit, "default", 100)
    frame, _ = _get_forecast_filtered(
        "historical",
        store_id=store_id,
        item_id=item_id,
        limit=limit_count,
    )
    frame = frame.copy()
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
    selected, _ = _get_forecast_filtered(
        request.forecast_type,
        store_id=request.store_id,
        item_id=request.item_id,
    )
    selected = selected.sort_values("date")
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
