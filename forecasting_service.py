"""Reusable XGBoost artifacts and forecasts from validated daily sales CSVs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_BUNDLE_PATH = MODEL_DIR / "recursive_xgboost.joblib"
ACTIVE_MODEL_PATH = MODEL_DIR / "active_recursive_xgboost.json"
FEATURE_CONFIG_PATH = ROOT / "data" / "model_ready" / "feature_config.json"
SALES_FILE = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
PRICES_FILE = ROOT / "data" / "raw" / "sell_prices.csv"
HISTORY_PATH = ROOT / "data" / "user_data" / "sales_history.parquet"
FORECAST_DIR = ROOT / "data" / "user_data" / "forecasts"

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
SERIES_ID_PATTERN = re.compile(
    r"^(?P<category>[^_]+)_(?P<department>[^_]+)_(?P<item>[^_]+)_"
    r"(?P<state>[A-Z]{2})_(?P<store_number>[0-9]+)_evaluation$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_names() -> list[str]:
    if not FEATURE_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Feature configuration not found: {FEATURE_CONFIG_PATH}")
    config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
    features = config.get("FEATURES")
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) for name in features
    ):
        raise ValueError("Feature configuration must contain a non-empty FEATURES list.")
    if len(features) != len(set(features)):
        raise ValueError("Feature configuration contains duplicate feature names.")
    if not DYNAMIC_FEATURES.issubset(features):
        raise ValueError("Feature configuration is missing required recursive features.")
    return features


def save_model_bundle(
    model: xgb.Booster,
    output_path: Path = MODEL_BUNDLE_PATH,
    *,
    training_date_utc: str | None,
    training_end_day: str,
    data_source: str,
    data_version: str | None = None,
    training_data_end_date: str | None = None,
    algorithm: str = "XGBoost (count:poisson)",
    evaluation_metrics: dict[str, Any] | None = None,
    activate: bool = False,
) -> dict[str, Any]:
    features = _feature_names()
    if model.num_features() != len(features):
        raise ValueError(
            f"Model has {model.num_features()} features; configuration has {len(features)}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_version = hashlib.sha256(model.save_raw(raw_format="json")).hexdigest()[:12]
    metadata = {
        "model_name": "recursive_xgboost",
        "algorithm": algorithm,
        "model_version": model_version,
        "training_date_utc": training_date_utc,
        "training_end_day": training_end_day,
        "training_data_end_date": training_data_end_date,
        "data_source": data_source,
        "data_version": data_version,
        "evaluation_metrics": evaluation_metrics,
        "feature_schema_version": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12],
        "feature_count": len(features),
        "training_rounds": model.num_boosted_rounds(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(
        {"model": model, "features": features, "metadata": metadata},
        temporary_path,
    )
    temporary_path.replace(output_path)
    metadata["artifact_sha256"] = _sha256(output_path)
    if activate:
        active_path = output_path.parent / ACTIVE_MODEL_PATH.name
        pointer_tmp = active_path.with_suffix(".json.tmp")
        pointer_tmp.write_text(
            json.dumps({"model_bundle_path": str(output_path.resolve())}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        pointer_tmp.replace(active_path)
    return metadata


def package_existing_model(
    model_path: Path,
    output_path: Path = MODEL_BUNDLE_PATH,
) -> dict[str, Any]:
    """Wrap an already-trained native XGBoost model; this does not train a model."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved XGBoost model not found: {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))

    # Look for corresponding training metadata if available
    training_date_utc = None
    data_source = f"Existing trained model artifact: {model_path.name}"
    data_version = None
    training_data_end_date = "2016-05-22"
    algorithm = "XGBoost (count:poisson)"

    # Check companion metadata file (e.g. *.metadata.json or job history)
    companion_meta = model_path.with_name(model_path.stem + ".metadata.json")
    if companion_meta.is_file():
        try:
            cmd = json.loads(companion_meta.read_text(encoding="utf-8"))
            training_date_utc = cmd.get("created_at_utc")
            data_version = cmd.get("data_version")
            data_source = f"M5 retail dataset (training cutoff: {cmd.get('training_cutoff', 'd_1941')})"
        except Exception:
            pass
    elif "d1941" in model_path.name:
        # Check if latest job has metadata for this model
        latest_job_file = ROOT / "data" / "model_ready" / "training_jobs" / "latest.json"
        if latest_job_file.is_file():
            try:
                lj = json.loads(latest_job_file.read_text(encoding="utf-8"))
                if lj.get("status") == "completed":
                    mm = lj.get("model_metadata", {})
                    training_date_utc = mm.get("created_at_utc") or lj.get("finished_at_utc")
                    data_version = mm.get("data_version")
                    data_source = "M5 retail dataset (sales_train_evaluation.csv through d_1941)"
            except Exception:
                pass

    # Read evaluation metrics from rolling backtest if available
    evaluation_metrics = None
    backtest_file = ROOT / "data" / "model_ready" / "rolling_backtest_metrics.csv"
    if backtest_file.is_file():
        try:
            bdf = pd.read_csv(backtest_file)
            xgb_metrics = bdf[bdf["method"] == "XGBoost"]
            if not xgb_metrics.empty:
                evaluation_metrics = {
                    "mean_mae": float(round(xgb_metrics["MAE"].mean(), 4)),
                    "mean_rmse": float(round(xgb_metrics["RMSE"].mean(), 4)),
                    "mean_wape_pct": float(round(xgb_metrics["WAPE_pct"].mean(), 2)),
                    "mean_bias": float(round(xgb_metrics["Bias"].mean(), 4)),
                    "backtest_periods": len(xgb_metrics),
                }
        except Exception:
            pass

    return save_model_bundle(
        model,
        output_path,
        training_date_utc=training_date_utc,
        training_end_day="d_1941",
        data_source=data_source,
        data_version=data_version,
        training_data_end_date=training_data_end_date,
        algorithm=algorithm,
        evaluation_metrics=evaluation_metrics,
        activate=True,
    )


def load_model_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is None or path == MODEL_BUNDLE_PATH:
        target_path = MODEL_BUNDLE_PATH
        if ACTIVE_MODEL_PATH.is_file():
            try:
                pointer = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
                active_str = pointer.get("model_bundle_path")
                if active_str and Path(active_str).is_file():
                    target_path = Path(active_str)
            except Exception:
                target_path = MODEL_BUNDLE_PATH
    else:
        target_path = path

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Reusable model bundle not found: {target_path}. Package the existing trained "
            "model with `python .\\forecasting_service.py package-existing`."
        )
    bundle = joblib.load(target_path)
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must contain a dictionary.")
    model = bundle.get("model")
    features = bundle.get("features")
    metadata = bundle.get("metadata")
    if not isinstance(model, xgb.Booster):
        raise ValueError("Model bundle does not contain a valid XGBoost Booster.")
    if features != _feature_names() or model.num_features() != len(features):
        raise ValueError("Saved model and current feature schema do not match.")
    if not isinstance(metadata, dict) or not metadata.get("model_version"):
        raise ValueError("Model bundle is missing version metadata.")
    return bundle


def validate_sales_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "date", "sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns id,date,sales; missing: "
            + ", ".join(sorted(missing))
        )
    allowed = required | {"sell_price"}
    unexpected = set(frame.columns).difference(allowed)
    if unexpected:
        raise ValueError(
            "Unsupported CSV columns: " + ", ".join(sorted(unexpected))
        )
    result = frame.copy()
    result["id"] = result["id"].astype("string").str.strip()
    if result["id"].isna().any() or result["id"].eq("").any():
        raise ValueError("CSV contains an empty series id.")
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ValueError("CSV contains an invalid date.")
    result["sales"] = pd.to_numeric(result["sales"], errors="coerce")
    sales = result["sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Sales values must be finite and non-negative.")
    if result.duplicated(["id", "date"]).any():
        raise ValueError("CSV contains duplicate id/date rows.")
    if "sell_price" not in result.columns:
        result["sell_price"] = np.nan
    else:
        raw_prices = result["sell_price"]
        result["sell_price"] = pd.to_numeric(raw_prices, errors="coerce")
        invalid_prices = (
            raw_prices.notna()
            & raw_prices.astype("string").str.strip().fillna("").ne("")
            & result["sell_price"].isna()
        )
        if invalid_prices.any():
            raise ValueError("Provided sell_price values must be numeric.")
        prices = result["sell_price"].dropna().to_numpy(dtype=np.float64)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Provided sell_price values must be finite and positive.")
    return result[["id", "date", "sales", "sell_price"]].sort_values(
        ["id", "date"]
    ).reset_index(drop=True)


def merge_sales_history(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    incoming = validate_sales_rows(incoming)
    if existing is None or existing.empty:
        return incoming
    previous = validate_sales_rows(existing)
    combined = pd.concat(
        [
            previous.assign(_source_order=0),
            incoming.assign(_source_order=1),
        ],
        ignore_index=True,
    ).sort_values(["id", "date", "_source_order"])
    combined = combined.drop_duplicates(["id", "date"], keep="last")
    combined["sell_price"] = combined.groupby("id", sort=False)["sell_price"].ffill()
    return combined.drop(columns="_source_order").reset_index(drop=True)


def load_m5_reference_context(
    series_ids: tuple[str, ...],
    training_end_day: str = "d_1941",
) -> dict[str, Any]:
    """Load M5-known series metadata and last-known prices for the requested IDs."""
    if not series_ids:
        return {"known_ids": set(), "series": {}, "prices": {}}
    if not SALES_FILE.is_file() or not PRICES_FILE.is_file() or not CALENDAR_FILE.is_file():
        return {
            "known_ids": set(),
            "series": {},
            "prices": {},
            "history": pd.DataFrame(
                columns=["id", "date", "sales", "sell_price"]
            ),
            "training_date": None,
            "training_end_day": training_end_day,
        }

    header = pd.read_csv(SALES_FILE, nrows=0).columns.tolist()
    day_columns = [column for column in header if column.startswith("d_")]
    requested_day = int(training_end_day.removeprefix("d_"))
    if requested_day > len(day_columns):
        raise ValueError(f"Training cutoff {training_end_day} exceeds available M5 sales.")
    history_days = day_columns[max(0, requested_day - 56):requested_day]
    sales_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = pd.read_csv(SALES_FILE, usecols=sales_columns + history_days)
    sales = sales.loc[sales["id"].isin(series_ids)].copy()

    calendar = pd.read_csv(CALENDAR_FILE)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    day_to_date = calendar.set_index("d")["date"]
    training_date = day_to_date.get(training_end_day)
    if training_date is None:
        raise ValueError(f"Calendar is missing training cutoff {training_end_day}.")
    cutoff_week = int(
        calendar.loc[calendar["d"].eq(training_end_day), "wm_yr_wk"].iloc[0]
    )
    history_rows: list[pd.DataFrame] = []
    for day in history_days:
        if day not in day_to_date.index:
            raise ValueError(f"Calendar is missing sales day {day}.")
        history_rows.append(pd.DataFrame({
            "id": sales["id"].astype(str),
            "date": day_to_date.loc[day],
            "sales": pd.to_numeric(sales[day], errors="raise").to_numpy(),
            "sell_price": np.nan,
        }))
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    series_info: dict[str, dict[str, str]] = {}
    pairs: set[tuple[str, str]] = set()
    for row in sales.itertuples(index=False):
        info = {
            "item_id": str(row.item_id),
            "store_id": str(row.store_id),
            "state_id": str(row.state_id),
        }
        series_info[str(row.id)] = info
        pairs.add((info["store_id"], info["item_id"]))

    price_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        PRICES_FILE,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        chunksize=500_000,
    ):
        mask = pd.MultiIndex.from_frame(chunk[["store_id", "item_id"]]).isin(pairs)
        if mask.any():
            selected = chunk.loc[mask & chunk["wm_yr_wk"].le(cutoff_week)]
            if not selected.empty:
                price_parts.append(selected)

    prices: dict[str, dict[str, float | int]] = {}
    if price_parts:
        price_frame = pd.concat(price_parts, ignore_index=True)
        for series_id, info in series_info.items():
            rows = price_frame.loc[
                price_frame["store_id"].eq(info["store_id"])
                & price_frame["item_id"].eq(info["item_id"])
            ].sort_values("wm_yr_wk")
            if rows.empty:
                continue
            unique_week = rows.drop_duplicates("wm_yr_wk", keep="last")
            current = float(unique_week.iloc[-1]["sell_price"])
            previous = (
                float(unique_week.iloc[-2]["sell_price"])
                if len(unique_week) > 1
                else current
            )
            median = float(rows["sell_price"].median())
            prices[series_id] = {
                "sell_price": current,
                "price_change": current - previous,
                "price_change_pct": (
                    100.0 * (current - previous) / previous if previous else 0.0
                ),
                "is_price_change": int(current != previous),
                "price_norm": current / median if median else 0.0,
                "price_available": 1,
            }
    return {
        "known_ids": set(series_info),
        "series": series_info,
        "prices": prices,
        "history": history,
        "training_date": pd.Timestamp(training_date),
        "training_end_day": training_end_day,
    }


def calendar_features_for_date(
    date: pd.Timestamp,
    calendar: pd.DataFrame,
) -> dict[str, float]:
    day = pd.Timestamp(date).normalize()
    required = {
        "date", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_name_2", "snap_CA", "snap_TX", "snap_WI",
    }
    if calendar.empty:
        return {
            "wm_yr_wk": float(day.isocalendar().year * 100 + day.isocalendar().week),
            "wday": float((day.dayofweek + 2) % 7 + 1),
            "month": float(day.month),
            "year": float(day.year),
            "day_of_week": float(day.dayofweek),
            "day_of_month": float(day.day),
            "week_of_year": float(day.isocalendar().week),
            "month_num": float(day.month),
            "year_num": float(day.year),
            "is_event": 0.0,
            "snap_CA": 0.0,
            "snap_TX": 0.0,
            "snap_WI": 0.0,
        }
    if not required.issubset(calendar.columns):
        raise ValueError("Calendar is missing fields required by the M5 feature pipeline.")
    parsed = calendar.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="raise").dt.normalize()
    exact = parsed.loc[parsed["date"].eq(day)]
    if len(exact) > 1:
        raise ValueError(f"Calendar contains duplicate rows for {day.date()}.")
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "wm_yr_wk": float(row["wm_yr_wk"]),
            "wday": float(row["wday"]),
            "month": float(row["month"]),
            "year": float(row["year"]),
            "day_of_week": float(day.dayofweek),
            "day_of_month": float(day.day),
            "week_of_year": float(day.isocalendar().week),
            "month_num": float(day.month),
            "year_num": float(day.year),
            "is_event": float(pd.notna(row["event_name_1"]) or pd.notna(row["event_name_2"])),
            "snap_CA": float(row["snap_CA"]),
            "snap_TX": float(row["snap_TX"]),
            "snap_WI": float(row["snap_WI"]),
        }

    last = parsed.sort_values("date").iloc[-1]
    last_date = pd.Timestamp(last["date"])
    last_week_start = last_date - pd.Timedelta(days=(int(last["wday"]) - 1) % 7)
    week_offset = max(0, int((day - last_week_start).days // 7))
    return {
        "wm_yr_wk": float(last["wm_yr_wk"]) + week_offset,
        "wday": float((day.dayofweek + 2) % 7 + 1),
        "month": float(day.month),
        "year": float(day.year),
        "day_of_week": float(day.dayofweek),
        "day_of_month": float(day.day),
        "week_of_year": float(day.isocalendar().week),
        "month_num": float(day.month),
        "year_num": float(day.year),
        "is_event": 0.0,
        "snap_CA": 0.0,
        "snap_TX": 0.0,
        "snap_WI": 0.0,
    }


def _trailing_contiguous_rows(group: pd.DataFrame, count: int) -> pd.DataFrame:
    recent = _trailing_contiguous_history(group, count)
    if len(recent) != count:
        return pd.DataFrame(columns=group.columns)
    return recent


def _trailing_contiguous_history(group: pd.DataFrame, max_count: int) -> pd.DataFrame:
    rows = group.sort_values("date").drop_duplicates("date", keep="last")
    if rows.empty:
        return rows
    dates = pd.DatetimeIndex(rows["date"]).to_numpy(dtype="datetime64[D]")
    breaks = np.flatnonzero(np.diff(dates) != np.timedelta64(1, "D"))
    run_start = int(breaks[-1] + 1) if breaks.size else 0
    return rows.iloc[run_start:].tail(max_count)


def _dynamic_feature_values(
    recent: np.ndarray,
    target_date: pd.Timestamp,
    last_sale: pd.Timestamp | None,
) -> dict[str, float]:
    return {
        "lag_1": float(recent[-1]),
        "lag_7": float(recent[-7]),
        "lag_14": float(recent[-14]),
        "lag_28": float(recent[-28]),
        "lag_56": float(recent[0]),
        "roll_mean_7": float(recent[-7:].mean()),
        "roll_std_7": float(recent[-7:].std(ddof=1)),
        "roll_mean_28": float(recent[-28:].mean()),
        "roll_std_28": float(recent[-28:].std(ddof=1)),
        "zero_ratio_7": float((recent[-7:] == 0).mean()),
        "zero_ratio_28": float((recent[-28:] == 0).mean()),
        "days_since_last_sale": (
            float((target_date - last_sale).days) if last_sale is not None else -1.0
        ),
        "no_previous_sale": float(last_sale is None),
    }


def _infer_state_id(series_id: str) -> str | None:
    """Try to extract a US state code from a series ID.

    Accepts M5 format (…_CA_1_evaluation, …_TX_2…) and any ID that embeds a
    two-letter state code preceded by an underscore.
    """
    match = SERIES_ID_PATTERN.match(series_id)
    if match:
        return match.group("state")
    # Fallback: scan tokens for a known state abbreviation
    for token in series_id.split("_"):
        if token in {"CA", "TX", "WI"}:
            return token
    return None


def generate_forecast(
    history: pd.DataFrame,
    bundle: dict[str, Any],
    reference: dict[str, Any],
    calendar: pd.DataFrame,
    *,
    horizon: int = 28,
    minimum_current_date: pd.Timestamp | None = None,
    allow_historical: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate recursive XGBoost forecasts for every compatible uploaded series.

    Each series is forecasted independently using the saved XGBoost model.
    Series with fewer than 56 history days use NaN for lag_56 (passed as
    missing to XGBoost).  Prediction errors are raised immediately; no silent
    fallback to a constant mean is performed.

    Forecasts start immediately after the latest observed sales date in the
    uploaded data. If the upload represents historical data older than
    minimum_current_date, the forecast is clearly categorized as
    'historical_forecast' unless allow_historical is False.

    Args:
        history: Validated sales history (id, date, sales, sell_price columns).
        bundle: Loaded model bundle from load_model_bundle().
        reference: M5 reference context from load_m5_reference_context().
        calendar: M5 calendar DataFrame (may be empty for dates beyond M5).
        horizon: Number of future days to predict (default 28).
        minimum_current_date: Optional cutoff date for determining if a forecast
            is current vs historical.
        allow_historical: If True (default), permits generating forecasts for
            historical date ranges. If False and data is older than
            minimum_current_date, raises ValueError.

    Returns:
        Tuple of (forecast DataFrame, validated history DataFrame).
    """
    rows = validate_sales_rows(history)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")

    # Use the per-series latest date; align all series to the global maximum so
    # that series with slightly different end dates are still accepted (the
    # forecast starts the day after the latest observed date overall).
    latest_dates = rows.groupby("id")["date"].max()
    as_of = pd.Timestamp(latest_dates.max()).normalize()

    # Reject future-dated observations beyond current wall-clock date plus margin
    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    if as_of > tomorrow:
        raise ValueError(
            f"Uploaded sales contain future dates up to {as_of.date()}. "
            "Sales observations must not be dated in the future."
        )

    is_historical = False
    if minimum_current_date is not None:
        min_date = pd.Timestamp(minimum_current_date).normalize()
        if as_of < min_date:
            if not allow_historical:
                raise ValueError(
                    f"Latest uploaded sales are from {as_of.date()}, which is before "
                    f"{min_date.date()}. Upload recent sales data for a genuine current forecast."
                )
            is_historical = True

    features = bundle["features"]
    model: xgb.Booster = bundle["model"]
    metadata = bundle["metadata"]

    if hasattr(model, "num_features") and callable(model.num_features):
        if model.num_features() != len(features):
            raise ValueError(
                f"Model has {model.num_features()} features but feature schema has "
                f"{len(features)}. Re-package or retrain the model."
            )

    ids = sorted(rows["id"].astype(str).unique())
    grouped = {series_id: frame for series_id, frame in rows.groupby("id", sort=False)}
    states: dict[str, dict[str, Any]] = {}
    model_ids: list[str] = []
    skipped: list[str] = []

    for series_id in ids:
        group = grouped[series_id].sort_values("date")
        last28 = _trailing_contiguous_rows(group, 28)
        if last28.empty:
            skipped.append(series_id)
            continue

        available_history = _trailing_contiguous_history(group, 56)
        sales_window = np.full(56, np.nan, dtype=np.float32)
        sales_history_values = available_history["sales"].to_numpy(dtype=np.float32)
        sales_window[-len(sales_history_values):] = sales_history_values
        model_ids.append(series_id)

        sold = group.loc[group["sales"].gt(0), "date"]
        last_sale = pd.Timestamp(sold.max()) if not sold.empty else None

        price_state = dict(reference["prices"].get(series_id, {}))
        uploaded_prices = group["sell_price"].dropna()
        if not uploaded_prices.empty:
            current_price = float(uploaded_prices.iloc[-1])
            previous_price = float(price_state.get("sell_price", current_price))
            median_price = (
                current_price / float(price_state["price_norm"])
                if price_state.get("price_norm")
                else current_price
            )
            price_state.update({
                "sell_price": current_price,
                "price_change": current_price - previous_price,
                "price_change_pct": (
                    100.0 * (current_price - previous_price) / previous_price
                    if previous_price
                    else 0.0
                ),
                "is_price_change": int(current_price != previous_price),
                "price_norm": current_price / median_price if median_price else 1.0,
                "price_available": 1,
            })

        defaults = {
            "sell_price": 0.0,
            "price_change": 0.0,
            "price_change_pct": 0.0,
            "is_price_change": 0.0,
            "price_norm": 0.0,
            "price_available": 0.0,
        }
        defaults.update(price_state)

        state_id = (
            reference.get("series", {}).get(series_id, {}).get("state_id")
            or _infer_state_id(series_id)
        )

        states[series_id] = {
            "sales": sales_window.copy(),
            "method": "recursive_xgboost",
            "last_sale": last_sale,
            "price": defaults,
            "state_id": state_id,
        }

    if skipped:
        raise ValueError(
            f"{len(skipped)} series need at least 28 consecutive daily observations "
            f"to generate recursive XGBoost features: {', '.join(skipped[:5])}"
            + (" …" if len(skipped) > 5 else "")
        )

    snap_columns = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    forecast_rows: list[dict[str, Any]] = []

    for offset in range(1, horizon + 1):
        target_date = as_of + pd.Timedelta(days=offset)
        calendar_row = calendar_features_for_date(target_date, calendar)

        feature_rows: list[dict[str, float]] = []
        for series_id in model_ids:
            state = states[series_id]
            recent = state["sales"]
            values: dict[str, float] = {
                name: float(value) for name, value in calendar_row.items()
                if name not in {"snap_CA", "snap_TX", "snap_WI"}
            }
            snap_column = snap_columns.get(state["state_id"])
            values["snap"] = (
                float(calendar_row[snap_column]) if snap_column is not None else 0.0
            )
            values.update({name: float(v) for name, v in state["price"].items()})
            values.update(
                _dynamic_feature_values(recent, target_date, state["last_sale"])
            )
            feature_rows.append(values)

        matrix = pd.DataFrame(feature_rows, index=model_ids)[features]

        unsupported_missing = [
            col for col in matrix.columns
            if col != "lag_56" and matrix[col].isna().any()
        ]
        if unsupported_missing:
            raise ValueError(
                f"XGBoost feature preparation failed on day {offset} "
                f"({target_date.date()}): missing values in "
                + ", ".join(unsupported_missing)
                + ". Check uploaded sales and price data."
            )

        feature_values = matrix.to_numpy(dtype=np.float32)
        if np.isinf(feature_values).any():
            raise ValueError(
                f"XGBoost feature preparation failed on day {offset} "
                f"({target_date.date()}): infinite values found."
            )

        try:
            raw_preds = np.asarray(
                model.predict(xgb.DMatrix(feature_values, missing=np.nan)),
                dtype=np.float64,
            ).reshape(-1)
        except (xgb.core.XGBoostError, ValueError, TypeError) as exc:
            raise ValueError(
                f"XGBoost prediction failed for {len(model_ids)} series "
                f"on {target_date.date()}: {exc}"
            ) from exc

        raw_preds = np.maximum(raw_preds, 0.0)
        if len(raw_preds) != len(model_ids) or not np.isfinite(raw_preds).all():
            raise ValueError(
                f"XGBoost returned {len(raw_preds)} predictions for "
                f"{len(model_ids)} series on {target_date.date()}; "
                "expected equal counts with all-finite values."
            )

        predictions: dict[str, float] = dict(zip(model_ids, raw_preds.tolist()))

        for series_id in model_ids:
            state = states[series_id]
            value = predictions[series_id]
            state["sales"] = np.roll(state["sales"], -1)
            state["sales"][-1] = np.float32(value)
            if value > 0.5:
                state["last_sale"] = target_date
            forecast_rows.append({
                "id": series_id,
                "date": target_date,
                "predicted_sales": float(value),
                "forecast_method": state["method"],
                "model_version": metadata["model_version"],
                "forecast_type": "historical" if is_historical else "current",
            })

    forecast = pd.DataFrame(forecast_rows)
    return forecast, rows


def prepare_uploaded_training_data(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    reference: dict[str, Any],
    *,
    minimum_days: int = 29,
    max_series: int = 10_000,
    examples_per_series: int = 28,
    minimum_rows: int = 10,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build bounded, leakage-aware training examples from recent uploaded history.

    Supports configurable dataset sizes down to 29 days (28 history days + 1 target day).
    For training windows with fewer than 56 history days, lag_56 is passed as NaN
    (which XGBoost handles natively). Never fabricates data.
    """
    rows = validate_sales_rows(history)
    if minimum_days < 29:
        raise ValueError(
            f"minimum_days must be at least 29 (needs 28 history days for lag_28/rolling features plus 1 target day); got {minimum_days}."
        )
    if max_series < 1 or examples_per_series < 1:
        raise ValueError("max_series and examples_per_series must be at least 1.")
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be at least 1.")

    features = _feature_names()
    ids = sorted(rows["id"].astype(str).unique())
    if len(ids) > max_series:
        ids = sorted(
            ids,
            key=lambda series_id: hashlib.sha256(
                series_id.encode("utf-8")
            ).hexdigest(),
        )[:max_series]

    calendar_cache: dict[pd.Timestamp, dict[str, float]] = {}
    feature_rows: list[dict[str, float]] = []
    labels: list[float] = []
    eligible_series = 0
    short_series: list[tuple[str, int]] = []

    for series_id in ids:
        group = rows.loc[rows["id"].eq(series_id)].sort_values("date")
        recent_history = _trailing_contiguous_rows(group, len(group))
        if len(recent_history) < minimum_days:
            short_series.append((series_id, len(recent_history)))
            continue

        eligible_series += 1
        sales = recent_history["sales"].to_numpy(dtype=np.float32)
        dates = pd.DatetimeIndex(recent_history["date"])
        prices = recent_history["sell_price"].to_numpy(dtype=np.float64)

        # Target range: from index 28 (where we have 28 preceding days: 0..27) up to len(recent_history)
        first_target = max(28, len(recent_history) - examples_per_series)

        series_state = reference.get("series", {}).get(series_id, {})
        state_id = series_state.get("state_id") or _infer_state_id(series_id)
        snap_columns = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

        for target_index in range(first_target, len(recent_history)):
            hist_slice = sales[:target_index]
            if len(hist_slice) < 28:
                continue

            # Build 56-slot window: if fewer than 56 history days, slots without history remain NaN
            recent = np.full(56, np.nan, dtype=np.float32)
            recent[-len(hist_slice[-56:]):] = hist_slice[-56:]

            target_date = pd.Timestamp(dates[target_index])
            if target_date not in calendar_cache:
                calendar_cache[target_date] = calendar_features_for_date(
                    target_date,
                    calendar,
                )
            calendar_row = calendar_cache[target_date]
            values = {
                name: float(value)
                for name, value in calendar_row.items()
                if name not in {"snap_CA", "snap_TX", "snap_WI"}
            }
            snap_column = snap_columns.get(state_id)
            values["snap"] = (
                float(calendar_row[snap_column]) if snap_column is not None else 0.0
            )
            prior_sales = recent_history.iloc[:target_index]
            sold_dates = prior_sales.loc[prior_sales["sales"].gt(0), "date"]
            last_sale = (
                pd.Timestamp(sold_dates.iloc[-1]) if not sold_dates.empty else None
            )
            values.update(_dynamic_feature_values(recent, target_date, last_sale))

            known_prices = prices[:target_index + 1]
            finite_prices = known_prices[np.isfinite(known_prices)]
            current_price = (
                float(known_prices[-1])
                if np.isfinite(known_prices[-1])
                else (float(finite_prices[-1]) if finite_prices.size else 0.0)
            )
            prior_finite_prices = prices[:target_index]
            prior_finite_prices = prior_finite_prices[
                np.isfinite(prior_finite_prices)
            ]
            previous_price = (
                float(prior_finite_prices[-1])
                if prior_finite_prices.size
                else current_price
            )
            median_price = (
                float(np.median(finite_prices)) if finite_prices.size else 0.0
            )
            reference_price = reference.get("prices", {}).get(series_id, {})
            if reference_price.get("price_norm") and current_price:
                median_price = current_price / float(reference_price["price_norm"])
            values.update({
                "sell_price": current_price,
                "price_change": current_price - previous_price,
                "price_change_pct": (
                    100.0 * (current_price - previous_price) / previous_price
                    if previous_price
                    else 0.0
                ),
                "is_price_change": float(current_price != previous_price),
                "price_norm": current_price / median_price if median_price else 0.0,
                "price_available": float(bool(finite_prices.size)),
            })
            feature_rows.append({name: values[name] for name in features})
            labels.append(float(sales[target_index]))

    summary = {
        "series_count": len(ids),
        "eligible_series": eligible_series,
        "training_rows": len(labels),
        "minimum_days": minimum_days,
        "max_series": max_series,
        "examples_per_series": examples_per_series,
        "training_data_end_date": rows["date"].max().strftime("%Y-%m-%d"),
    }
    if eligible_series == 0:
        details = ", ".join(f"{sid} ({cnt} days)" for sid, cnt in short_series[:5])
        raise ValueError(
            f"No series meet the minimum requirement of {minimum_days} consecutive days. "
            f"Short series: {details}."
        )
    if len(labels) < minimum_rows:
        raise ValueError(
            f"Not enough training examples generated ({len(labels):,}) from {eligible_series:,} eligible series; "
            f"require at least {minimum_rows:,} examples. Each series needs at least {minimum_days} consecutive days."
        )
    matrix = pd.DataFrame(feature_rows, columns=features)
    target = np.asarray(labels, dtype=np.float32)

    # Check for unsupported NaNs: only lag_56 is allowed to be NaN
    unsupported_missing = [
        col for col in matrix.columns
        if col != "lag_56" and matrix[col].isna().any()
    ]
    if unsupported_missing:
        raise ValueError(
            f"Uploaded training features contain unexpected missing values in: {', '.join(unsupported_missing)}."
        )
    if np.isinf(matrix.to_numpy(dtype=np.float32)).any():
        raise ValueError("Uploaded training features contain non-finite values.")
    return matrix, target, summary


def train_uploaded_history(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    reference: dict[str, Any],
    *,
    minimum_days: int = 29,
    minimum_rows: int = 10,
    num_boost_round: int = 150,
) -> dict[str, Any]:
    """Explicitly retrain and activate a reusable model from saved uploaded history.

    Preserves the previous active model until the newly trained model is validated
    and successfully saved.
    """
    matrix, target, summary = prepare_uploaded_training_data(
        history,
        calendar,
        reference,
        minimum_days=minimum_days,
        minimum_rows=minimum_rows,
    )
    if num_boost_round < 1:
        raise ValueError("Training must use at least one boosting round.")
    thread_count = max(1, min(8, os.cpu_count() or 2))
    dtrain = xgb.DMatrix(matrix.to_numpy(dtype=np.float32), label=target, missing=np.nan)
    model = xgb.train(
        {
            "objective": "count:poisson",
            "eval_metric": "mae",
            "tree_method": "hist",
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "max_bin": 32,
            "nthread": thread_count,
            "seed": 42,
        },
        dtrain,
        num_boost_round=num_boost_round,
    )
    clean_history = validate_sales_rows(history)
    data_digest = hashlib.sha256(
        clean_history[["id", "date", "sales", "sell_price"]]
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()
    output_path = MODEL_DIR / f"recursive_xgboost_uploaded_{data_digest[:12]}.joblib"

    # Evaluate training fit
    preds = model.predict(dtrain)
    mae_fit = float(np.mean(np.abs(preds - target)))

    saved_meta = save_model_bundle(
        model,
        output_path,
        training_date_utc=datetime.now(timezone.utc).isoformat(),
        training_end_day="uploaded_data",
        data_source="Saved user-uploaded sales history",
        data_version=data_digest,
        training_data_end_date=summary["training_data_end_date"],
        algorithm="XGBoost (count:poisson)",
        evaluation_metrics={"training_fit_mae": round(mae_fit, 4), "training_examples": len(target)},
        activate=True,
    )
    return saved_meta | summary


def save_forecast_result(
    forecast: pd.DataFrame,
    actual_history: pd.DataFrame,
    *,
    model_metadata: dict[str, Any],
    source_name: str,
) -> dict[str, str]:
    """Persist validated output and its model/data provenance."""
    if forecast.empty:
        raise ValueError("Cannot save an empty forecast.")
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    upload_digest = hashlib.sha256(
        actual_history[["id", "date", "sales"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    forecast_id = upload_digest[:12]
    forecast_path = FORECAST_DIR / f"forecast_{forecast_id}.parquet"
    metadata_path = FORECAST_DIR / f"forecast_{forecast_id}.json"
    history_path = HISTORY_PATH
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_history = history_path.with_suffix(".parquet.tmp")
    actual_history.to_parquet(temporary_history, index=False)
    temporary_history.replace(history_path)
    forecast.to_parquet(forecast_path, index=False)
    forecast_type = "historical" if (forecast["date"].max() < pd.Timestamp.now().normalize()) else "current"
    details = {
        "forecast_id": forecast_id,
        "forecast_type": forecast_type,
        "model_version": model_metadata["model_version"],
        "algorithm": model_metadata.get("algorithm", "XGBoost (count:poisson)"),
        "training_date_utc": model_metadata.get("training_date_utc"),
        "training_end_day": model_metadata.get("training_end_day"),
        "training_data_end_date": model_metadata.get("training_data_end_date"),
        "model_data_source": model_metadata.get("data_source"),
        "model_data_version": model_metadata.get("data_version"),
        "evaluation_metrics": model_metadata.get("evaluation_metrics"),
        "data_source": source_name,
        "upload_history_sha256": upload_digest,
        "as_of_date": (
            pd.Timestamp(forecast["date"].min()) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "forecast_start": forecast["date"].min().strftime("%Y-%m-%d"),
        "forecast_end": forecast["date"].max().strftime("%Y-%m-%d"),
        "forecast_rows": int(len(forecast)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_path": str(forecast_path),
    }
    metadata_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    latest_path = FORECAST_DIR / "latest.json"
    latest_path.write_text(
        json.dumps({"metadata_path": str(metadata_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return details


def load_latest_forecast() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    latest_path = FORECAST_DIR / "latest.json"
    if not latest_path.is_file() or not HISTORY_PATH.is_file():
        return None
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    metadata_path = Path(pointer["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    forecast_path = Path(metadata["forecast_path"])
    if not forecast_path.is_file():
        raise FileNotFoundError(f"Saved forecast artifact not found: {forecast_path}")
    return (
        pd.read_parquet(forecast_path),
        pd.read_parquet(HISTORY_PATH),
        metadata,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=("package-existing",),
        help="Package an existing trained XGBoost model without retraining.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "final_xgboost_d1941.json",
    )
    parser.add_argument("--output", type=Path, default=MODEL_BUNDLE_PATH)
    args = parser.parse_args()
    metadata = package_existing_model(args.model, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
