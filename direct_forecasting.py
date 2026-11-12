"""Horizon-aware direct XGBoost forecasts for days 1 through 28."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
SALES_FILE = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
FEATURE_FILE = ROOT / "data" / "model_ready" / "m5_model_features.parquet"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "model_ready"
MODEL_PATH = MODEL_DIR / "direct_xgboost.json"
MODEL_INFO_PATH = MODEL_DIR / "direct_xgboost_metadata.json"
FORECAST_PATH = OUTPUT_DIR / "direct_forecast_d1942_d1969.parquet"
DIRECT_FEATURE_FILE = OUTPUT_DIR / "direct_training_examples.parquet"
FORECAST_HORIZON = 28
BOOST_ROUNDS = 150
MAX_BIN = 32
BATCH_SIZE = 250_000

BASE_FEATURES = [
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month_num",
    "year_num",
    "sell_price",
    "price_change",
    "price_change_pct",
    "is_price_change",
    "price_available",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "roll_std_28",
    "zero_ratio_7",
    "zero_ratio_28",
    "days_since_last_sale",
    "no_previous_sale",
    "is_event",
    "snap",
]
ENCODED_FEATURES = [
    "category_code",
    "department_code",
    "item_code",
    "store_code",
    "state_code",
    "horizon",
]
DIRECT_FEATURES = BASE_FEATURES + ENCODED_FEATURES
DYNAMIC_FEATURES = {
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "roll_std_28",
    "zero_ratio_7",
    "zero_ratio_28",
    "days_since_last_sale",
    "no_previous_sale",
}
ID_METADATA = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class ParquetIterator(xgb.DataIter):
    def __init__(
        self,
        path: Path,
        feature_names: list[str],
        target: str,
        cache_prefix: str,
    ) -> None:
        self.path = path
        self.feature_names = feature_names
        self.target = target
        self.dataset = ds.dataset(path, format="parquet")
        super().__init__(cache_prefix=cache_prefix)
        self.reset()

    def next(self, input_data: Any) -> bool:
        batch = next(self._batches, None)
        if batch is None:
            return False
        frame = batch.to_pandas()
        input_data(
            data=frame[self.feature_names].to_numpy(dtype=np.float32),
            label=frame[self.target].to_numpy(dtype=np.float32),
        )
        return True

    def reset(self) -> None:
        self._batches = iter(
            self.dataset.scanner(
                columns=self.feature_names + [self.target],
                batch_size=BATCH_SIZE,
            ).to_batches()
        )


def _day_number(day: str) -> int:
    if not day.startswith("d_"):
        raise ValueError(f"Expected an M5 day label such as d_1941, got {day!r}.")
    try:
        value = int(day[2:])
    except ValueError as exc:
        raise ValueError(f"Invalid M5 day label: {day!r}") from exc
    if value < 1:
        raise ValueError("M5 day numbers start at 1.")
    return value


def _load_inputs(
    sales_path: Path = SALES_FILE,
    calendar_path: Path = CALENDAR_FILE,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    if not sales_path.is_file():
        raise FileNotFoundError(f"Sales data not found: {sales_path}")
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar data not found: {calendar_path}")
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    day_columns = [column for column in header if column.startswith("d_")]
    expected = [f"d_{day}" for day in range(1, len(day_columns) + 1)]
    if day_columns != expected:
        raise ValueError("Sales data must have consecutive d_1...d_N columns.")
    sales = pd.read_csv(sales_path, usecols=ID_METADATA + day_columns)
    if sales["id"].duplicated().any():
        raise ValueError("Sales data contains duplicate series IDs.")
    values = sales[day_columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales values must be finite and non-negative.")
    calendar = pd.read_csv(calendar_path)
    required = {
        "d",
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    }
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(f"Calendar is missing required fields: {sorted(missing)}")
    if calendar["d"].duplicated().any():
        raise ValueError("Calendar contains duplicate day keys.")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["day_of_month"] = calendar["date"].dt.day
    calendar["week_of_year"] = (
        calendar["date"].dt.isocalendar().week.astype("int16")
    )
    calendar["month_num"] = calendar["date"].dt.month
    calendar["year_num"] = calendar["date"].dt.year
    calendar["is_event"] = (
        calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
    ).astype(np.int8)
    calendar = calendar.set_index("d", drop=False)
    if len(day_columns) > len(calendar):
        raise ValueError("Calendar does not cover every sales day.")
    return sales, values, day_columns, calendar


def _identifier_codes(sales: pd.DataFrame) -> dict[str, np.ndarray]:
    id_parts = sales["id"].str.extract(
        r"^(?P<category>[^_]+)_(?P<department>[^_]+)_(?P<sku>[^_]+)_"
        r"(?P<state>[A-Z]{2})_(?P<store_number>[0-9]+)_evaluation$"
    )
    if id_parts.isna().any(axis=None):
        raise ValueError("At least one M5 series ID has an unsupported format.")

    def codes(values: pd.Series) -> np.ndarray:
        return pd.Categorical(values).codes.astype(np.float32)

    return {
        "category_code": codes(id_parts["category"]),
        "department_code": codes(id_parts["department"]),
        "item_code": codes(sales["item_id"]),
        "store_code": codes(sales["store_id"]),
        "state_code": codes(sales["state_id"]),
    }


def _last_sale_days(values: np.ndarray) -> np.ndarray:
    series_count, day_count = values.shape
    result = np.zeros((series_count, day_count), dtype=np.int16)
    last_seen = np.zeros(series_count, dtype=np.int16)
    for day_index in range(day_count):
        sold = values[:, day_index] > 0
        last_seen[sold] = day_index + 1
        result[:, day_index] = last_seen
    return result


def _get_origin_prices(
    origins: list[int],
    series_ids: list[str],
    feature_path: Path = FEATURE_FILE,
) -> dict[int, pd.DataFrame]:
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Model-ready feature data is required for origin prices: {feature_path}"
        )
    dataset = ds.dataset(feature_path, format="parquet")
    day_names = [f"d_{origin}" for origin in origins]
    table = dataset.to_table(
        columns=[
            "id",
            "d",
            "sell_price",
            "price_change",
            "price_change_pct",
            "is_price_change",
            "price_available",
        ],
        filter=ds.field("d").isin(day_names),
    )
    frame = table.to_pandas()
    result: dict[int, pd.DataFrame] = {}
    for origin in origins:
        day_frame = (
            frame.loc[frame["d"] == f"d_{origin}"]
            .set_index("id")
            .reindex(series_ids)
        )
        if day_frame.index.hasnans or day_frame.isna().all(axis=1).any():
            raise ValueError(f"Origin price features are missing for d_{origin}.")
        result[origin] = day_frame
    return result


def _origin_dynamic_features(
    sales_values: np.ndarray,
    last_sale_days: np.ndarray,
    origin: int,
) -> dict[str, np.ndarray]:
    recent = sales_values[:, origin - 56:origin]
    if recent.shape[1] != 56:
        raise ValueError(f"At least 56 days of history are required for d_{origin}.")
    last_sale = last_sale_days[:, origin - 1]
    has_previous_sale = last_sale > 0
    days_since = np.full(len(last_sale), -1, dtype=np.float32)
    days_since[has_previous_sale] = origin - last_sale[has_previous_sale]
    return {
        "lag_1": recent[:, -1],
        "lag_7": recent[:, -7],
        "lag_14": recent[:, -14],
        "lag_28": recent[:, -28],
        "lag_56": recent[:, 0],
        "roll_mean_7": recent[:, -7:].mean(axis=1),
        "roll_std_7": recent[:, -7:].std(axis=1, ddof=1),
        "roll_mean_28": recent[:, -28:].mean(axis=1),
        "roll_std_28": recent[:, -28:].std(axis=1, ddof=1),
        "zero_ratio_7": (recent[:, -7:] == 0).mean(axis=1),
        "zero_ratio_28": (recent[:, -28:] == 0).mean(axis=1),
        "days_since_last_sale": days_since,
        "no_previous_sale": (~has_previous_sale).astype(np.int8),
    }


def _origin_price_features(
    price: pd.DataFrame,
    series_count: int,
) -> dict[str, np.ndarray]:
    if len(price) != series_count:
        raise ValueError("Origin price feature count does not match the sales series.")
    return {
        "sell_price": price["sell_price"].fillna(0).to_numpy(dtype=np.float32),
        "price_change": price["price_change"].fillna(0).to_numpy(dtype=np.float32),
        "price_change_pct": price["price_change_pct"].fillna(0).to_numpy(dtype=np.float32),
        "is_price_change": price["is_price_change"].fillna(0).to_numpy(dtype=np.int8),
        "price_available": price["price_available"].fillna(0).to_numpy(dtype=np.int8),
    }


def _features_for_horizon(
    origin: int,
    horizon: int,
    dynamic: dict[str, np.ndarray],
    price: dict[str, np.ndarray],
    codes: dict[str, np.ndarray],
    calendar: pd.DataFrame,
    state_ids: list[str],
) -> dict[str, np.ndarray]:
    series_count = len(state_ids)
    target_day = origin + horizon
    day_name = f"d_{target_day}"
    if day_name not in calendar.index:
        raise ValueError(f"Calendar does not include {day_name}.")
    day = calendar.loc[day_name]
    if isinstance(day, pd.DataFrame):
        raise ValueError(f"Calendar has duplicate rows for {day_name}.")
    snap_for_state = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    state_column_by_code = np.asarray(
        [snap_for_state.get(state, "") for state in state_ids],
        dtype=object,
    )
    snap = np.zeros(series_count, dtype=np.float32)
    for column in ("snap_CA", "snap_TX", "snap_WI"):
        mask = state_column_by_code == column
        if mask.any():
            snap[mask] = float(day[column])
    fields = {
        name: np.full(series_count, day[name], dtype=np.float32)
        for name in (
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "month_num",
            "year_num",
            "is_event",
        )
    }
    fields["snap"] = snap
    for name, values in dynamic.items():
        if name == "days_since_last_sale":
            values = np.where(values >= 0, values + horizon, values)
        fields[name] = values
    fields.update(price)
    fields.update(codes)
    fields["horizon"] = np.full(series_count, horizon, dtype=np.float32)
    return fields


def _make_origin_list(training_end_day: int, step: int = 28) -> list[int]:
    if step < 1:
        raise ValueError("Origin step must be at least one day.")
    max_origin = training_end_day - FORECAST_HORIZON
    if max_origin < 56:
        raise ValueError("Training end day must leave 56 history days plus 28 target days.")
    origins = list(range(56, max_origin + 1, step))
    if not origins or origins[-1] != max_origin:
        origins.append(max_origin)
    return sorted(set(origins))


def build_training_examples(
    training_end_day: int,
    output_path: Path = DIRECT_FEATURE_FILE,
    origin_step_days: int = 28,
) -> tuple[Path, dict[str, Any]]:
    """Write direct multi-horizon training rows using only origin-known inputs."""
    sales, values, day_columns, calendar = _load_inputs()
    if training_end_day >= len(day_columns):
        raise ValueError(
            f"Training end d_{training_end_day} exceeds available sales d_{len(day_columns)}."
        )
    ids = sales["id"].astype(str).tolist()
    state_ids = sales["state_id"].astype(str).tolist()
    codes = _identifier_codes(sales)
    last_sale = _last_sale_days(values[:, :training_end_day])
    origins = _make_origin_list(training_end_day, origin_step_days)
    origin_prices = _get_origin_prices(origins, ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for origin in origins:
            dynamic = _origin_dynamic_features(values, last_sale, origin)
            price = _origin_price_features(origin_prices[origin], len(ids))
            for horizon in range(1, FORECAST_HORIZON + 1):
                feature_values = _features_for_horizon(
                    origin,
                    horizon,
                    dynamic,
                    price,
                    codes,
                    calendar,
                    state_ids,
                )
                frame = pd.DataFrame(feature_values, columns=DIRECT_FEATURES)
                frame["sales"] = values[:, origin + horizon - 1]
                frame["forecast_origin_day"] = np.int16(origin)
                frame["horizon"] = np.int8(horizon)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=100_000)
                row_count += len(frame)
            print(f"Built direct training examples for origin d_{origin}.", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        raise ValueError("No direct training examples were generated.")
    info = {
        "training_end_day": training_end_day,
        "origin_days": origins,
        "origin_step_days": origin_step_days,
        "forecast_horizon_days": FORECAST_HORIZON,
        "training_rows": row_count,
        "feature_names": DIRECT_FEATURES,
        "price_policy": (
            "Carry the last price known at the forecast origin through the horizon; "
            "future realized price changes are not used."
        ),
    }
    return output_path, info


class RoundProgress(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int = BOOST_ROUNDS) -> None:
        self.total_rounds = total_rounds

    def after_iteration(
        self,
        _model: xgb.Booster,
        epoch: int,
        _evals_log: dict[str, Any],
    ) -> bool:
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == self.total_rounds:
            print(f"Completed direct model round {epoch + 1}/{self.total_rounds}.", flush=True)
        return False


def train_direct_model(
    training_end_day: int,
    num_boost_round: int = BOOST_ROUNDS,
    origin_step_days: int = 28,
) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "xgb_cache").mkdir(parents=True, exist_ok=True)
    output_path, training_info = build_training_examples(
        training_end_day=training_end_day,
        output_path=DIRECT_FEATURE_FILE,
        origin_step_days=origin_step_days,
    )
    dataset = ds.dataset(output_path, format="parquet")
    row_count = dataset.count_rows()
    with tempfile.TemporaryDirectory(
        prefix="direct_xgb_",
        dir=MODEL_DIR / "xgb_cache",
    ) as cache_dir:
        iterator = ParquetIterator(
            output_path,
            DIRECT_FEATURES,
            "sales",
            str(Path(cache_dir) / "direct_train"),
        )
        matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=MAX_BIN)
        params = {
            "objective": "count:poisson",
            "eval_metric": "mae",
            "tree_method": "hist",
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "max_bin": MAX_BIN,
            "nthread": max(1, min(8, os.cpu_count() or 2)),
            "seed": 42,
        }
        model = xgb.train(
            params=params,
            dtrain=matrix,
            num_boost_round=num_boost_round,
            callbacks=[RoundProgress(num_boost_round)],
        )
        del matrix, iterator
        gc.collect()

    model.feature_names = DIRECT_FEATURES
    model.save_model(str(MODEL_PATH))
    schema_payload = json.dumps(DIRECT_FEATURES, separators=(",", ":")).encode("utf-8")
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    data_version = _input_data_version()
    model_info = {
        "model_name": "direct_xgboost",
        "model_version": model_hash[:12],
        "model_sha256": model_hash,
        "data_version": data_version,
        "feature_schema_version": hashlib.sha256(schema_payload).hexdigest()[:12],
        "feature_names": DIRECT_FEATURES,
        "training_rows": row_count,
        "training_end_day": training_end_day,
        "num_boost_round": num_boost_round,
        "training_configuration": params,
        "forecast_horizon_days": FORECAST_HORIZON,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": training_info,
    }
    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Direct XGBoost model saved to {MODEL_PATH}", flush=True)
    return MODEL_PATH, MODEL_INFO_PATH


def _input_data_version() -> str:
    digest = hashlib.sha256()
    for path in (SALES_FILE, CALENDAR_FILE, FEATURE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required versioned input is missing: {path}")
        file_digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(f"{path.name}:{file_digest.hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest()


def generate_direct_forecast(
    origin_day: int,
    model_path: Path = MODEL_PATH,
    output_path: Path | None = None,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Direct XGBoost model not found: {model_path}. Train it first."
        )
    sales, values, day_columns, calendar = _load_inputs()
    if origin_day > len(day_columns):
        raise ValueError("Forecast origin exceeds the last day with actual sales.")
    if origin_day < 56:
        raise ValueError("Forecast origin requires at least 56 days of history.")
    if origin_day + FORECAST_HORIZON > len(calendar):
        raise ValueError("Calendar does not cover the requested 28-day forecast.")
    ids = sales["id"].astype(str).tolist()
    state_ids = sales["state_id"].astype(str).tolist()
    model = xgb.Booster()
    model.load_model(str(model_path))
    if model.num_features() != len(DIRECT_FEATURES):
        raise ValueError(
            f"Model expects {model.num_features()} features; "
            f"direct schema provides {len(DIRECT_FEATURES)}."
        )
    codes = _identifier_codes(sales)
    last_sale = _last_sale_days(values[:, :origin_day])
    dynamic = _origin_dynamic_features(values, last_sale, origin_day)
    origin_prices = _get_origin_prices([origin_day], ids)[origin_day]
    price = _origin_price_features(origin_prices, len(ids))
    metadata = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    outputs = []
    for horizon in range(1, FORECAST_HORIZON + 1):
        frame = pd.DataFrame(
            _features_for_horizon(
                origin_day,
                horizon,
                dynamic,
                price,
                codes,
                calendar,
                state_ids,
            ),
            columns=DIRECT_FEATURES,
        )
        prediction = np.maximum(
            model.predict(
                xgb.DMatrix(
                    frame.to_numpy(dtype=np.float32),
                    feature_names=DIRECT_FEATURES,
                )
            ),
            0.0,
        )
        target_day = origin_day + horizon
        outputs.append(pd.DataFrame({
            "id": ids,
            "store_id": sales["store_id"],
            "item_id": sales["item_id"],
            "forecast_origin_day": f"d_{origin_day}",
            "forecast_date_day": f"d_{target_day}",
            "forecast_date": calendar.loc[f"d_{target_day}", "date"],
            "horizon": horizon,
            "raw_prediction": prediction.astype(np.float32),
            "point_forecast": prediction.astype(np.float32),
            "model_version": metadata["model_version"],
            "data_version": metadata["data_version"],
            "feature_schema_version": metadata["feature_schema_version"],
        }))
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["store_id", "item_id", "horizon"]).any():
        raise ValueError("Generated direct forecast has duplicate series/horizon rows.")
    if output_path is None:
        output_path = (
            OUTPUT_DIR
            / f"direct_forecast_d{origin_day + 1}_d{origin_day + FORECAST_HORIZON}.parquet"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved direct forecast to {output_path}", flush=True)
    return output_path


def evaluate_direct_forecast(
    forecast_path: Path,
    output_path: Path | None = None,
) -> Path:
    if not forecast_path.is_file():
        raise FileNotFoundError(f"Direct forecast file not found: {forecast_path}")
    forecast = pd.read_parquet(forecast_path)
    required = {"id", "forecast_date_day", "horizon", "point_forecast"}
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"Direct forecast is missing columns: {sorted(missing)}")
    sales, values, day_columns, _ = _load_inputs()
    id_positions = pd.Series(np.arange(len(sales)), index=sales["id"].astype(str))
    positions = forecast["id"].map(id_positions)
    if positions.isna().any():
        raise ValueError("Direct forecast contains IDs not found in the sales data.")
    forecast = forecast.copy()
    forecast["series_position"] = positions.to_numpy(dtype=np.int64)
    day_numbers = forecast["forecast_date_day"].map(_day_number)
    if day_numbers.max() > len(day_columns):
        raise ValueError("Cannot evaluate forecast days beyond available actual sales.")
    forecast["actual_sales"] = [
        values[position, day_number - 1]
        for position, day_number in zip(
            forecast["series_position"],
            day_numbers,
        )
    ]
    forecast["error"] = forecast["point_forecast"] - forecast["actual_sales"]

    def metrics(rows: pd.DataFrame, horizon: int) -> dict[str, float | int]:
        actual_total = float(rows["actual_sales"].abs().sum())
        if actual_total == 0:
            raise ValueError("WAPE cannot be calculated when total actual sales are zero.")
        return {
            "horizon": horizon,
            "MAE": float(rows["error"].abs().mean()),
            "RMSE": float(np.sqrt(rows["error"].pow(2).mean())),
            "WAPE_pct": float(100 * rows["error"].abs().sum() / actual_total),
            "Bias": float(rows["error"].mean()),
            "row_count": int(len(rows)),
        }

    rows = [metrics(forecast, 0)]
    rows.extend(
        metrics(group, int(horizon))
        for horizon, group in forecast.groupby("horizon", sort=True)
    )
    metrics_frame = pd.DataFrame(rows)
    if output_path is None:
        output_path = OUTPUT_DIR / "direct_evaluation_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_path, index=False)
    return output_path


def run_direct_backtest(
    num_boost_round: int = BOOST_ROUNDS,
    origin_step_days: int = 28,
) -> Path:
    folds = [(1857, 1885), (1885, 1913), (1913, 1941)]
    fold_results: list[dict[str, Any]] = []
    for training_end, validation_end in folds:
        validation_start = training_end + 1
        train_direct_model(
            training_end,
            num_boost_round=num_boost_round,
            origin_step_days=origin_step_days,
        )
        forecast_path = OUTPUT_DIR / (
            f"direct_forecast_d{validation_start}_d{validation_end}.parquet"
        )
        generate_direct_forecast(training_end, output_path=forecast_path)
        metrics_path = OUTPUT_DIR / (
            f"direct_metrics_d{validation_start}_d{validation_end}.csv"
        )
        evaluate_direct_forecast(forecast_path, metrics_path)
        overall = pd.read_csv(metrics_path).loc[lambda rows: rows["horizon"] == 0]
        metadata = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        row = overall.iloc[0]
        fold_results.append({
            "training_cutoff": f"d_{training_end}",
            "validation_start": f"d_{validation_start}",
            "validation_end": f"d_{validation_end}",
            "direct_model_version": metadata["model_version"],
            "direct_data_version": metadata["data_version"],
            "direct_MAE": float(row["MAE"]),
            "direct_RMSE": float(row["RMSE"]),
            "direct_WAPE_pct": float(row["WAPE_pct"]),
            "direct_Bias": float(row["Bias"]),
        })
        print(
            f"Completed direct fold d_{validation_start}–d_{validation_end}.",
            flush=True,
        )

    results = pd.DataFrame(fold_results)
    recursive_path = OUTPUT_DIR / "rolling_backtest_metrics.csv"
    if recursive_path.is_file():
        recursive = pd.read_csv(recursive_path)
        recursive = recursive.loc[
            recursive["method"].isin(["XGBoost", "28-day mean baseline"])
        ]
        recursive = recursive.pivot(
            index=["training_cutoff", "validation_start", "validation_end"],
            columns="method",
            values=["MAE", "RMSE", "WAPE_pct", "Bias"],
        )
        recursive.columns = [
            f"{method.lower().replace(' ', '_').replace('-', '_')}_{metric}"
            for metric, method in recursive.columns
        ]
        results = results.merge(
            recursive.reset_index(),
            on=["training_cutoff", "validation_start", "validation_end"],
            how="left",
            validate="one_to_one",
        )
    output_path = OUTPUT_DIR / "direct_rolling_backtest_metrics.csv"
    results.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--training-end-day", type=int, default=1913)
    train_parser.add_argument("--rounds", type=int, default=BOOST_ROUNDS)
    train_parser.add_argument("--origin-step-days", type=int, default=28)
    forecast_parser = subparsers.add_parser("forecast")
    forecast_parser.add_argument("--origin-day", type=int, default=1941)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--forecast", type=Path, default=FORECAST_PATH)
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "direct_evaluation_metrics.csv",
    )
    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--rounds", type=int, default=BOOST_ROUNDS)
    backtest_parser.add_argument("--origin-step-days", type=int, default=28)
    args = parser.parse_args()
    if args.command == "train":
        paths = train_direct_model(
            args.training_end_day,
            num_boost_round=args.rounds,
            origin_step_days=args.origin_step_days,
        )
        print(f"Model: {paths[0]}")
        print(f"Metadata: {paths[1]}")
    elif args.command == "forecast":
        print(generate_direct_forecast(args.origin_day))
    elif args.command == "evaluate":
        print(evaluate_direct_forecast(args.forecast, args.output))
    elif args.command == "backtest":
        print(
            run_direct_backtest(
                num_boost_round=args.rounds,
                origin_step_days=args.origin_step_days,
            )
        )


if __name__ == "__main__":
    main()
