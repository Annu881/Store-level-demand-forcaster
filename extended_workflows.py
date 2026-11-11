"""Rolling evaluation and future forecasting without rerunning the notebook."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
FEATURE_DATA = ROOT / "data" / "model_ready" / "m5_model_features.parquet"
FEATURE_CONFIG = ROOT / "data" / "model_ready" / "feature_config.json"
SALES_FILE = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
PRICES_FILE = ROOT / "data" / "raw" / "sell_prices.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "data" / "model_ready"

BOOST_ROUNDS = 150
MAX_BIN = 32
BATCH_ROWS = 250_000
SALE_THRESHOLD = 0.5
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


class ParquetTrainingIterator(xgb.DataIter):
    def __init__(
        self,
        dataset: ds.Dataset,
        columns: list[str],
        row_filter: Any,
        batch_size: int,
        cache_prefix: str,
    ) -> None:
        self.dataset = dataset
        self.columns = columns
        self.row_filter = row_filter
        self.batch_size = batch_size
        super().__init__(cache_prefix=cache_prefix)
        self.reset()

    def next(self, input_data: Any) -> bool:
        batch = next(self._batches, None)
        if batch is None:
            return False
        frame = batch.to_pandas()
        input_data(
            data=frame[self.columns[:-1]].to_numpy(dtype=np.float32),
            label=frame[self.columns[-1]].to_numpy(dtype=np.float32),
        )
        return True

    def reset(self) -> None:
        self._batches = iter(
            self.dataset.scanner(
                columns=self.columns,
                filter=self.row_filter,
                batch_size=self.batch_size,
            ).to_batches()
        )


class TrainingProgress(xgb.callback.TrainingCallback):
    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: dict[str, Any],
    ) -> bool:
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == BOOST_ROUNDS:
            print(f"Completed boosting round {epoch + 1}/{BOOST_ROUNDS}.", flush=True)
        return False


def load_config() -> tuple[list[str], str]:
    config = json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))
    features = config["FEATURES"]
    target = config["TARGET"]
    if len(features) != len(set(features)):
        raise ValueError("Feature configuration contains duplicate feature names.")
    if not DYNAMIC_FEATURES.issubset(features):
        raise ValueError(
            f"Feature configuration is missing dynamic features: "
            f"{sorted(DYNAMIC_FEATURES.difference(features))}"
        )
    return features, target


def load_feature_dataset() -> ds.Dataset:
    if not FEATURE_DATA.is_file():
        raise FileNotFoundError(f"Feature data was not found: {FEATURE_DATA}")
    return ds.dataset(FEATURE_DATA, format="parquet")


def train_model(dataset: ds.Dataset, features: list[str], target: str, end_day: int) -> xgb.Booster:
    training_days = [f"d_{day}" for day in range(1, end_day + 1)]
    row_filter = ds.field("d").isin(training_days)
    columns = features + [target]
    row_count = dataset.count_rows(filter=row_filter)
    if row_count == 0:
        raise ValueError(f"No training rows found through d_{end_day}.")

    thread_count = max(1, min(8, os.cpu_count() or 2))
    parameters = {
        "objective": "count:poisson",
        "eval_metric": "mae",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "max_bin": MAX_BIN,
        "nthread": thread_count,
        "seed": 42,
    }

    print(
        f"Training through d_{end_day}: {row_count:,} rows, "
        f"{BOOST_ROUNDS} boosting rounds.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"xgb_train_d{end_day}_",
        dir=MODEL_DIR / "xgb_cache",
    ) as cache_directory:
        iterator = ParquetTrainingIterator(
            dataset=dataset,
            columns=columns,
            row_filter=row_filter,
            batch_size=BATCH_ROWS,
            cache_prefix=str(Path(cache_directory) / "train"),
        )
        matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=MAX_BIN)
        model = xgb.train(
            params=parameters,
            dtrain=matrix,
            num_boost_round=BOOST_ROUNDS,
            callbacks=[TrainingProgress()],
        )
        del matrix, iterator
        gc.collect()
    return model


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    actual_total = float(np.abs(actual).sum())
    if actual_total == 0:
        raise ValueError("WAPE cannot be calculated when total actual sales are zero.")
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE_pct": float(100.0 * np.abs(error).sum() / actual_total),
        "Bias": float(error.mean()),
    }


def recursive_validation(
    model: xgb.Booster,
    dataset: ds.Dataset,
    features: list[str],
    start_day: int,
    end_day: int,
) -> tuple[dict[str, float], dict[str, float]]:
    validation_days = [f"d_{day}" for day in range(start_day, end_day + 1)]
    validation_filter = ds.field("d").isin(validation_days)
    validation = dataset.to_table(
        columns=["id", "d", "sales"] + features,
        filter=validation_filter,
    ).to_pandas()
    if validation.empty:
        raise ValueError(
            f"No validation rows found for d_{start_day} through d_{end_day}."
        )
    validation = validation.sort_values(["d", "id"]).reset_index(drop=True)
    first_day = validation.loc[validation["d"] == validation_days[0]].set_index("id")
    series_ids = first_day.index.tolist()
    if not series_ids or validation["id"].nunique() != len(series_ids):
        raise ValueError("Validation period does not have a consistent series set.")
    expected_ids = set(series_ids)
    if any(
        set(group["id"]) != expected_ids
        for _, group in validation.groupby("d", sort=False)
    ):
        raise ValueError("Every validation day must contain the same series set.")

    history_days = [f"d_{day}" for day in range(start_day - 56, start_day)]
    history_filter = ds.field("d").isin(history_days)
    history_frame = dataset.to_table(
        columns=["id", "d", "sales"],
        filter=history_filter,
    ).to_pandas()
    history = (
        history_frame.pivot(index="id", columns="d", values="sales")
        .reindex(index=series_ids, columns=history_days)
    )
    if history.isna().any().any():
        raise ValueError("Validation history is missing one or more required sales values.")

    days_since = pd.to_numeric(first_day["days_since_last_sale"], errors="raise")
    no_previous = pd.to_numeric(first_day["no_previous_sale"], errors="raise")
    if days_since.isna().any() or not no_previous.isin([0, 1]).all():
        raise ValueError("Invalid sale-history state at the validation cutoff.")
    if not (days_since.loc[no_previous.eq(1)] == -1).all():
        raise ValueError("Never-sold series must use days_since_last_sale=-1.")

    last_sale_day = pd.Series(np.nan, index=series_ids, dtype=np.float64)
    has_previous_sale = no_previous.eq(0)
    last_sale_day.loc[has_previous_sale] = (
        start_day - days_since.loc[has_previous_sale].astype(np.float64)
    )
    actual_values: list[np.ndarray] = []
    prediction_values: list[np.ndarray] = []

    for day in range(start_day, end_day + 1):
        day_name = f"d_{day}"
        day_frame = (
            validation.loc[validation["d"] == day_name]
            .set_index("id")
            .reindex(series_ids)
            .copy()
        )
        recent_days = [f"d_{value}" for value in range(day - 56, day)]
        recent = history.reindex(index=series_ids, columns=recent_days).to_numpy(
            dtype=np.float32
        )
        last_sale_today = last_sale_day.reindex(series_ids)
        has_sale = last_sale_today.notna().to_numpy()
        days_since_today = np.full(len(series_ids), -1.0, dtype=np.float32)
        last_sale_values = last_sale_today.to_numpy(dtype=np.float64)
        days_since_today[has_sale] = day - last_sale_values[has_sale]

        dynamic_values = {
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
            "days_since_last_sale": days_since_today,
            "no_previous_sale": (~has_sale).astype(np.int8),
        }
        for name, values in dynamic_values.items():
            day_frame[name] = values

        actual = pd.to_numeric(day_frame["sales"], errors="raise").to_numpy(
            dtype=np.float64
        )
        matrix = xgb.DMatrix(day_frame[features].to_numpy(dtype=np.float32))
        prediction = np.maximum(
            np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1), 0.0
        )
        if len(prediction) != len(series_ids) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid model predictions on {day_name}.")

        actual_values.append(actual)
        prediction_values.append(prediction)
        history[day_name] = prediction
        sale_mask = prediction > SALE_THRESHOLD
        last_sale_day.loc[np.asarray(series_ids)[sale_mask]] = day

    actual_all = np.concatenate(actual_values)
    predicted_all = np.concatenate(prediction_values)
    baseline_values = np.empty_like(actual_all)
    actual_by_day = actual_all.reshape(end_day - start_day + 1, len(series_ids))
    predicted_by_day = predicted_all.reshape(end_day - start_day + 1, len(series_ids))

    baseline_days = [f"d_{day}" for day in range(start_day - 28, start_day)]
    baseline_filter = ds.field("d").isin(baseline_days)
    baseline_frame = dataset.to_table(
        columns=["id", "d", "sales"],
        filter=baseline_filter,
    ).to_pandas()
    baseline_history = (
        baseline_frame.pivot(index="id", columns="d", values="sales")
        .reindex(index=series_ids, columns=baseline_days)
    )
    if baseline_history.isna().any().any():
        raise ValueError("Baseline history is missing one or more required sales values.")
    baseline_per_series = baseline_history.mean(axis=1).to_numpy(dtype=np.float64)
    baseline_values[:] = np.tile(baseline_per_series, end_day - start_day + 1)

    model_metrics = metric_row(actual_all, predicted_all)
    baseline_metrics = metric_row(actual_all, baseline_values)
    return model_metrics, baseline_metrics


def run_backtests(dataset: ds.Dataset, features: list[str], target: str) -> Path:
    folds = [
        (1857, 1858, 1885),
        (1885, 1886, 1913),
        (1913, 1914, 1941),
    ]
    existing_model_path = MODEL_DIR / "final_xgboost.json"
    results: list[dict[str, Any]] = []
    result_path = RESULT_DIR / "rolling_backtest_metrics.csv"

    for cutoff, start_day, end_day in folds:
        if cutoff == 1913 and existing_model_path.is_file():
            model = xgb.Booster()
            model.load_model(str(existing_model_path))
            if model.num_features() != len(features) or model.num_boosted_rounds() != BOOST_ROUNDS:
                print("Saved fold-3 model has unexpected dimensions; retraining it.")
                model = train_model(dataset, features, target, cutoff)
            else:
                print(
                    "Using the saved model trained through d_1913 for the "
                    "d_1914–d_1941 fold.",
                    flush=True,
                )
        else:
            model = train_model(dataset, features, target, cutoff)

        model_metrics, baseline_metrics = recursive_validation(
            model=model,
            dataset=dataset,
            features=features,
            start_day=start_day,
            end_day=end_day,
        )
        for method, metrics in (
            ("XGBoost", model_metrics),
            ("28-day mean baseline", baseline_metrics),
        ):
            results.append({
                "training_cutoff": f"d_{cutoff}",
                "validation_start": f"d_{start_day}",
                "validation_end": f"d_{end_day}",
                "method": method,
                **metrics,
            })
        pd.DataFrame(results).to_csv(result_path, index=False)
        print(
            f"Completed validation d_{start_day}–d_{end_day}. "
            f"Results saved to {result_path}",
            flush=True,
        )
        del model
        gc.collect()
    return result_path


def read_future_prices(future_weeks: set[int]) -> pd.DataFrame:
    parts = []
    columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    for chunk in pd.read_csv(PRICES_FILE, usecols=columns, chunksize=500_000):
        selected = chunk.loc[chunk["wm_yr_wk"].isin(future_weeks)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("No sell-price records exist for the requested forecast weeks.")
    prices = pd.concat(parts, ignore_index=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Future sell-price data has duplicate store/item/week rows.")
    return prices


def run_future_forecast(
    dataset: ds.Dataset,
    features: list[str],
    target: str,
) -> tuple[Path, Path]:
    forecast_start, forecast_end = 1942, 1969
    latest_actual_day = forecast_start - 1
    if not (SALES_FILE.is_file() and CALENDAR_FILE.is_file() and PRICES_FILE.is_file()):
        raise FileNotFoundError("Raw sales, calendar, and sell-price CSVs are required.")

    model = train_model(dataset, features, target, latest_actual_day)
    model_path = MODEL_DIR / "final_xgboost_d1941.json"
    model.save_model(str(model_path))
    print(f"Future-forecast model saved to {model_path}", flush=True)

    calendar = pd.read_csv(CALENDAR_FILE)
    forecast_days = [f"d_{day}" for day in range(forecast_start, forecast_end + 1)]
    future_calendar = calendar.loc[calendar["d"].isin(forecast_days)].copy()
    if future_calendar["d"].nunique() != len(forecast_days):
        raise ValueError("Calendar does not contain every requested future day.")
    future_calendar["date"] = pd.to_datetime(future_calendar["date"])
    future_calendar["day_of_week"] = future_calendar["date"].dt.dayofweek
    future_calendar["day_of_month"] = future_calendar["date"].dt.day
    future_calendar["week_of_year"] = (
        future_calendar["date"].dt.isocalendar().week.astype("int16")
    )
    future_calendar["month_num"] = future_calendar["date"].dt.month
    future_calendar["year_num"] = future_calendar["date"].dt.year
    future_calendar["is_event"] = (
        future_calendar["event_name_1"].notna()
        | future_calendar["event_name_2"].notna()
    ).astype(np.int8)

    history_days = [f"d_{day}" for day in range(latest_actual_day - 55, latest_actual_day + 1)]
    metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = pd.read_csv(
        SALES_FILE,
        usecols=metadata_columns + history_days,
    )
    if sales["id"].duplicated().any():
        raise ValueError("Sales data has duplicate series IDs.")
    series_ids = sales["id"].tolist()
    history = sales.set_index("id")[history_days].reindex(series_ids)
    if history.isna().any().any():
        raise ValueError("Historical sales have missing values for future forecast series.")
    history = history.astype(np.float32)
    metadata = sales[metadata_columns].copy()

    final_day_features = dataset.to_table(
        columns=[
            "id",
            "days_since_last_sale",
            "no_previous_sale",
            "sell_price",
            "price_norm",
            "price_available",
        ],
        filter=ds.field("d") == f"d_{latest_actual_day}",
    ).to_pandas().set_index("id").reindex(series_ids)
    if final_day_features.isna().any().any():
        raise ValueError("Missing final-day features for one or more forecast series.")

    never_sold = final_day_features["no_previous_sale"].eq(1)
    last_sale_day = pd.Series(np.nan, index=series_ids, dtype=np.float64)
    previously_sold = ~never_sold
    last_sale_day.loc[previously_sold] = (
        latest_actual_day
        - final_day_features.loc[previously_sold, "days_since_last_sale"].astype(float)
    )
    final_actuals = sales.set_index("id")[f"d_{latest_actual_day}"].reindex(series_ids)
    last_sale_day.loc[final_actuals.gt(0)] = latest_actual_day

    current_price_at_cutoff = final_day_features["sell_price"].astype(float).copy()
    current_price_at_cutoff.loc[
        final_day_features["price_available"].eq(0)
    ] = np.nan
    price_norm_at_cutoff = final_day_features["price_norm"].astype(float)
    median_price = current_price_at_cutoff.div(price_norm_at_cutoff.replace(0, np.nan))

    future_weeks = set(future_calendar["wm_yr_wk"].astype(int))
    future_prices = read_future_prices(future_weeks)
    snap_columns = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    predictions: list[pd.DataFrame] = []
    previous_price = current_price_at_cutoff.reindex(series_ids).to_numpy(dtype=np.float64)

    for day in range(forecast_start, forecast_end + 1):
        day_name = f"d_{day}"
        calendar_row = future_calendar.loc[future_calendar["d"] == day_name]
        if len(calendar_row) != 1:
            raise ValueError(f"Expected one calendar row for {day_name}.")
        calendar_row = calendar_row.iloc[0]
        day_frame = metadata.copy()
        day_frame["id"] = series_ids
        day_frame["d"] = day_name
        day_frame["date"] = calendar_row["date"]
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
        ):
            day_frame[name] = calendar_row[name]
        snap_by_state = {
            state: calendar_row[column]
            for state, column in snap_columns.items()
        }
        unknown_states = sorted(set(day_frame["state_id"]).difference(snap_by_state))
        if unknown_states:
            raise ValueError(f"Unknown state codes in series metadata: {unknown_states}")
        day_frame["snap"] = day_frame["state_id"].map(snap_by_state)

        price_for_week = future_prices.loc[
            future_prices["wm_yr_wk"].eq(calendar_row["wm_yr_wk"]),
            ["store_id", "item_id", "sell_price"],
        ]
        day_frame = day_frame.merge(
            price_for_week,
            on=["store_id", "item_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        day_frame = day_frame.set_index("id").reindex(series_ids).reset_index()
        current_price = day_frame["sell_price"].to_numpy(dtype=np.float64)
        change = current_price - previous_price
        with np.errstate(divide="ignore", invalid="ignore"):
            change_pct = change / previous_price * 100.0
        day_frame["price_change"] = change
        day_frame["price_change_pct"] = change_pct
        day_frame["is_price_change"] = (change != 0).astype(np.int8)
        day_frame["price_norm"] = (
            day_frame["sell_price"].to_numpy(dtype=np.float64)
            / median_price.reindex(series_ids).to_numpy(dtype=np.float64)
        )
        day_frame["price_available"] = day_frame["sell_price"].notna().astype(np.int8)
        day_frame[
            ["sell_price", "price_change", "price_change_pct", "price_norm"]
        ] = day_frame[
            ["sell_price", "price_change", "price_change_pct", "price_norm"]
        ].fillna(0)

        recent_days = [f"d_{value}" for value in range(day - 56, day)]
        recent = history.reindex(index=series_ids, columns=recent_days).to_numpy(
            dtype=np.float32
        )
        last_sale_today = last_sale_day.reindex(series_ids)
        has_sale = last_sale_today.notna().to_numpy()
        days_since_today = np.full(len(series_ids), -1.0, dtype=np.float32)
        last_sale_values = last_sale_today.to_numpy(dtype=np.float64)
        days_since_today[has_sale] = day - last_sale_values[has_sale]
        dynamic_values = {
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
            "days_since_last_sale": days_since_today,
            "no_previous_sale": (~has_sale).astype(np.int8),
        }
        for name, values in dynamic_values.items():
            day_frame[name] = values

        matrix = xgb.DMatrix(day_frame[features].to_numpy(dtype=np.float32))
        predicted = np.maximum(
            np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1), 0.0
        )
        if len(predicted) != len(series_ids) or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid future predictions on {day_name}.")
        predictions.append(pd.DataFrame({
            "id": series_ids,
            "d": day_name,
            "date": calendar_row["date"],
            "predicted_sales": predicted.astype(np.float32),
        }))

        history[day_name] = predicted.astype(np.float32)
        sale_mask = predicted > SALE_THRESHOLD
        last_sale_day.loc[np.asarray(series_ids)[sale_mask]] = day
        previous_price = current_price
        print(f"Forecasted {day_name}", flush=True)

    output_path = RESULT_DIR / "future_forecast_d1942_d1969.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(output_path, index=False)
    del model
    gc.collect()
    return model_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=("backtest", "future", "all"),
        help="Run rolling backtests, a future forecast, or both.",
    )
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "xgb_cache").mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    features, target = load_config()
    dataset = load_feature_dataset()

    if args.task in ("backtest", "all"):
        result_path = run_backtests(dataset, features, target)
        print(f"Rolling backtest results: {result_path}")
    if args.task in ("future", "all"):
        model_path, forecast_path = run_future_forecast(dataset, features, target)
        print(f"Updated final model: {model_path}")
        print(f"Future forecasts: {forecast_path}")


if __name__ == "__main__":
    main()
