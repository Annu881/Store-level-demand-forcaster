"""Simple Streamlit dashboard for exploring saved demand forecasts."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

import forecasting_service
from inventory import ReplenishmentInputs, simulate_replenishment
import training_jobs
import api_client


ROOT = Path(__file__).resolve().parent
MODEL_READY = ROOT / "data" / "model_ready"
QUALITY_REPORT = ROOT / "data" / "quality" / "latest.json"
FORECAST_FILES = {
    "Past 28-day chart: estimates vs actual sales": (
        MODEL_READY / "final_forecast_d1914_d1941.parquet",
        True,
    ),
}
BACKTEST_FILE = MODEL_READY / "rolling_backtest_metrics.csv"
DIRECT_METRICS_FILE = MODEL_READY / "direct_evaluation_metrics.csv"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
ID_PATTERN = (
    r"^(?P<category>[^_]+)_(?P<department>[^_]+)_"
    r"(?P<item>[^_]+)_(?P<store>[^_]+_[^_]+)_evaluation$"
)

st.set_page_config(
    page_title="Store Demand Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f5f8fc;
        border: 1px solid #e6edf5;
        padding: 1rem 1.15rem;
        border-radius: 0.8rem;
    }
    .intro {color: #566579; font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_forecast(path_string: str, has_actuals: bool) -> pd.DataFrame:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Forecast data not found: {path}")
    frame = pd.read_parquet(path)
    if "point_forecast" in frame.columns:
        frame = frame.rename(
            columns={
                "forecast_date_day": "d",
                "point_forecast": "predicted_sales",
            }
        )
    required = {"id", "d", "predicted_sales"}
    if has_actuals:
        required.add("actual_sales")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    parsed = frame["id"].str.extract(ID_PATTERN)
    if parsed.isna().to_numpy().any():
        raise ValueError("Some forecast series IDs do not match the expected M5 format.")
    frame = pd.concat([frame, parsed], axis=1)
    if "date" not in frame.columns:
        if not CALENDAR_FILE.is_file():
            raise FileNotFoundError(f"Calendar data not found: {CALENDAR_FILE}")
        calendar = pd.read_csv(CALENDAR_FILE, usecols=["d", "date"])
        frame = frame.merge(calendar, on="d", how="left", validate="many_to_one")
        if frame["date"].isna().any():
            raise ValueError("Calendar is missing dates for one or more forecast days.")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner=False)
def load_quality_report(path_string: str) -> dict[str, object]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data-quality report not found: {path}. Run data_quality.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def active_model_bundle_path() -> Path:
    if forecasting_service.ACTIVE_MODEL_PATH.is_file():
        try:
            pointer = json.loads(
                forecasting_service.ACTIVE_MODEL_PATH.read_text(encoding="utf-8")
            )
            active_path = Path(pointer["model_bundle_path"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Active model pointer is invalid: {forecasting_service.ACTIVE_MODEL_PATH}"
            ) from exc
        if not active_path.is_file():
            raise FileNotFoundError(
                f"Active model bundle was not found: {active_path}"
            )
        return active_path

    job = training_jobs.read_latest_job()
    artifacts = job.get("artifacts", {}) if job else {}
    if job and job.get("status") == "completed" and isinstance(artifacts, dict):
        candidate = Path(str(artifacts.get("model_bundle_path", "")))
        if candidate.is_file():
            return candidate

    if forecasting_service.MODEL_BUNDLE_PATH.is_file():
        return forecasting_service.MODEL_BUNDLE_PATH

    legacy_model = ROOT / "models" / "final_xgboost_d1941.json"
    if not legacy_model.is_file():
        raise FileNotFoundError(
            "No saved recursive XGBoost model is available. Train or provide the "
            "model artifact before requesting a forecast."
        )
    forecasting_service.package_existing_model(legacy_model)
    return forecasting_service.MODEL_BUNDLE_PATH


@st.cache_resource(show_spinner=False)
def load_model_resource(path_string: str, modified_ns: int) -> dict[str, object]:
    del modified_ns
    return forecasting_service.load_model_bundle(Path(path_string))


@st.cache_data(show_spinner=False)
def load_m5_reference(
    series_ids: tuple[str, ...],
    training_end_day: str,
) -> dict[str, object]:
    return forecasting_service.load_m5_reference_context(
        series_ids,
        training_end_day,
    )


def load_saved_user_result() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    return forecasting_service.load_latest_forecast()


def _training_progress(status: dict[str, object], log_lines: list[str]) -> tuple[float, str]:
    state = str(status.get("status", ""))
    if state == "completed":
        return 1.0, "Training and forecasting complete"
    if state == "failed":
        return 1.0, "Training failed"

    round_progress = [
        (int(match.group(1)), int(match.group(2)))
        for line in log_lines
        if (match := re.search(r"Completed boosting round (\d+)/(\d+)", line))
    ]
    forecast_days = [
        int(match.group(1))
        for line in log_lines
        if (match := re.search(r"Forecasted d_(\d+)", line))
    ]
    if forecast_days:
        complete_days = min(28, len(set(forecast_days)))
        return 0.9 + 0.1 * complete_days / 28, (
            f"Generating future forecasts: {complete_days}/28 days"
        )
    if round_progress:
        completed_rounds, total_rounds = round_progress[-1]
        progress = 0.9 * completed_rounds / total_rounds
        return progress, (
            f"Training XGBoost: round {completed_rounds}/{total_rounds}"
        )
    if status.get("stage") == "preparing":
        return 0.02, "Preparing training data"
    return 0.0, "Starting training worker"


@st.fragment(run_every="5s")
def _training_status_panel() -> None:
    status = training_jobs.read_latest_job()
    if status is None:
        st.info("No training job has been started from this dashboard yet.")
        return

    job_id = str(status.get("job_id", "unknown"))
    read_tail = getattr(training_jobs, "read_log_tail", lambda path, lines=24: [])
    log_lines = read_tail(str(status.get("log_path", "")))
    progress, progress_label = _training_progress(status, log_lines)
    if status.get("status") in {"starting", "running"}:
        st.progress(progress, text=progress_label)
        st.info(
            "Training runs in the background. You can leave this page open; status "
            "refreshes every few seconds."
        )
        if st.button("⏹️ Cancel background training", key=f"cancel-job-{job_id}"):
            training_jobs.cancel_active_job()
            st.rerun(scope="app")
    elif status.get("status") == "completed":
        if st.session_state.get("cleared_training_cache_job") != job_id:
            st.cache_data.clear()
            st.session_state["cleared_training_cache_job"] = job_id
            st.rerun(scope="app")
        st.success("Training finished. The model and forecast have been saved.")
    elif status.get("status") == "failed":
        st.error(f"Training failed: {status.get('error', 'See the job log below.')}")
        if st.button("🔄 Clear status & Reset training lock", key=f"clear-lock-{job_id}"):
            training_jobs.cancel_active_job()
            st.rerun(scope="app")
    else:
        st.warning(f"Training job state: {status.get('status', 'unknown')}")

    st.caption(
        f"Job {job_id} · Started {status.get('started_at_utc', 'unknown')} · "
        f"Stage: {status.get('stage', 'unknown')}"
    )
    if status.get("status") == "completed":
        metadata = status.get("model_metadata", {})
        if isinstance(metadata, dict):
            st.write(
                f"Model version `{metadata.get('model_version', 'unknown')}` · "
                f"Data version `{str(metadata.get('data_version', 'unknown'))[:12]}` · "
                f"{metadata.get('forecast_row_count', 0):,} saved forecasts"
            )
        artifacts = status.get("artifacts", {})
        if isinstance(artifacts, dict):
            model_path = Path(str(artifacts["model_path"]))
            model_bundle_path = Path(
                str(artifacts.get("model_bundle_path", ""))
            )
            forecast_path = Path(str(artifacts["forecast_path"]))
            metadata_path = Path(str(artifacts["metadata_path"]))
            downloads = st.columns(4)
            if model_path.is_file():
                downloads[0].download_button(
                    "Download XGBoost model",
                    data=model_path.read_bytes(),
                    file_name=model_path.name,
                    mime="application/json",
                    key=f"download-model-{job_id}",
                )
            if forecast_path.is_file():
                downloads[1].download_button(
                    "Download forecast",
                    data=forecast_path.read_bytes(),
                    file_name=forecast_path.name,
                    mime="application/vnd.apache.parquet",
                    key=f"download-forecast-{job_id}",
                )
            if model_bundle_path.is_file():
                downloads[2].download_button(
                    "Download reusable joblib model",
                    data=model_bundle_path.read_bytes(),
                    file_name=model_bundle_path.name,
                    mime="application/octet-stream",
                    key=f"download-model-bundle-{job_id}",
                )
            if metadata_path.is_file():
                downloads[3].download_button(
                    "Download model details",
                    data=metadata_path.read_bytes(),
                    file_name=metadata_path.name,
                    mime="application/json",
                    key=f"download-metadata-{job_id}",
                )
    if log_lines:
        with st.expander("Recent training output"):
            st.code("\n".join(log_lines), language="text")


@st.cache_data(show_spinner=False)
def load_direct_importance(model_path: str, metadata_path: str) -> pd.DataFrame:
    import json
    import xgboost as xgb

    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    if not model_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Direct XGBoost model or metadata is unavailable.")
    model = xgb.Booster()
    model.load_model(str(model_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    scores = model.get_score(importance_type="gain")
    feature_names = metadata["feature_names"]
    rows = [
        {
            "Feature": (
                feature_names[int(feature[1:])]
                if feature.startswith("f") and feature[1:].isdigit()
                else feature
            ),
            "Gain importance": float(score),
        }
        for feature, score in scores.items()
    ]
    return pd.DataFrame(rows).sort_values("Gain importance", ascending=False)


@st.cache_data(show_spinner=False)
def load_backtest(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Backtest results not found: {path}")
    results = pd.read_csv(path)
    required = {
        "validation_start",
        "validation_end",
        "method",
        "MAE",
        "RMSE",
        "WAPE_pct",
        "Bias",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if not CALENDAR_FILE.is_file():
        raise FileNotFoundError(f"Calendar data not found: {CALENDAR_FILE}")
    calendar = pd.read_csv(CALENDAR_FILE, usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    date_lookup = calendar.set_index("d")["date"]
    results["test_period"] = results.apply(
        lambda row: format_date_range(
            date_lookup.loc[row["validation_start"]],
            date_lookup.loc[row["validation_end"]],
        ),
        axis=1,
    )
    results["method"] = results["method"].replace({
        "XGBoost": "Forecast model (XGBoost)",
        "28-day mean baseline": "Simple recent-average estimate",
    })
    return results


def parse_series_metadata(series_id: str) -> dict[str, str]:
    """Extract store, item, department, category, and state from any series ID.

    Supports:
    - Standard M5 evaluation format: CATEGORY_DEPT_ITEM_STATE_STORE_evaluation
    - Standard M5 format without suffix: CATEGORY_DEPT_ITEM_STATE_STORE (e.g. FOODS_1_001_CA_1)
    - Custom / generic delimited formats
    """
    cleaned = str(series_id).strip()
    # 1. Match M5 pattern with or without _evaluation
    m = re.match(
        r"^(?P<category>[^_]+)_(?P<department>[^_]+)_(?P<item>[^_]+)_(?P<store>[A-Z]{2}_[0-9]+)(?:_evaluation)?$",
        cleaned,
    )
    if m:
        d = m.groupdict()
        cat = d["category"]
        dept = f"{cat}_{d['department']}"
        item_id = f"{dept}_{d['item']}"
        store_id = d["store"]
        state_id = store_id.split("_")[0]
        return {
            "category": cat,
            "department": dept,
            "item": item_id,
            "store": store_id,
            "state": state_id,
        }

    # 2. General pattern: at least 3 parts, last two look like state_store or store code
    parts = cleaned.split("_")
    if len(parts) >= 4 and len(parts[-2]) == 2 and parts[-2].isalpha():
        # e.g. ITEM_NAME_CA_1
        store_id = f"{parts[-2]}_{parts[-1]}"
        item_id = "_".join(parts[:-2])
        return {
            "category": parts[0],
            "department": "_".join(parts[:2]) if len(parts) >= 5 else parts[0],
            "item": item_id,
            "store": store_id,
            "state": parts[-2],
        }

    if len(parts) >= 2:
        return {
            "category": parts[0],
            "department": parts[0],
            "item": "_".join(parts[:-1]),
            "store": parts[-1],
            "state": "Unknown",
        }

    return {
        "category": "General",
        "department": "General",
        "item": cleaned,
        "store": "Store_1",
        "state": "Unknown",
    }


def readable_series(series_id: str) -> str:
    meta = parse_series_metadata(series_id)
    cat_name = readable_category(meta["category"])
    store_desc = readable_store(meta["store"])
    return f"{cat_name} · Item {meta['item']} · {store_desc}"


def readable_store(store: str) -> str:
    parts = store.split("_", maxsplit=1)
    if len(parts) == 2:
        state, store_number = parts
        state_name = {
            "CA": "California",
            "TX": "Texas",
            "WI": "Wisconsin",
        }.get(state, state)
        return f"{state_name}, Store {store_number}"
    return f"Store {store}"


def readable_category(category: str) -> str:
    return {
        "FOODS": "Foods",
        "HOBBIES": "Hobbies",
        "HOUSEHOLD": "Household",
    }.get(category, category.title())


def format_date(value: pd.Timestamp) -> str:
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_text = start.strftime("%B %d").replace(" 0", " ")
    end_text = end.strftime("%B %d").replace(" 0", " ")
    if start.year == end.year:
        return f"{start_text}–{end_text}, {end.year}"
    return f"{format_date(start)}–{format_date(end)}"


def format_units(value: float) -> str:
    return f"{value:,.0f} units"


def describe_bias(value: float) -> str:
    if abs(value) < 0.005:
        return "About even"
    direction = "Underpredicted" if value < 0 else "Overpredicted"
    return f"{direction} by {abs(value):.2f} units/day"


st.title("📦 Store Demand Forecast")
st.markdown(
    '<p class="intro">See estimated product sales by store, compare estimates with '
    "past sales, and learn how reliable the estimates were.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Explore")
    page = st.radio(
        "Choose a view",
        [
            "Explore sales estimates",
            "Past accuracy",
            "Replenishment simulator",
            "Data quality",
            "Train and save M5 forecast",
            "About this tool",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    api_online = api_client.default_client.is_healthy()
    if api_online:
        st.caption("🟢 **Backend API**: Connected (`http://127.0.0.1:8000`)")
    else:
        st.caption("🟡 **Backend API**: Offline (Local mode)")

if page == "Explore sales estimates":
    st.subheader("Upload recent sales and create a future forecast")
    st.markdown(
        "Upload a CSV with one row per product/store/day. Required columns are "
        "`id,date,sales`; `sell_price` is optional. Re-uploaded `id,date` rows replace "
        "older saved values. The forecast starts after the newest observed sales date."
    )
    st.download_button(
        "Download CSV header template",
        data="id,date,sales,sell_price\n",
        file_name="daily_sales_template.csv",
        mime="text/csv",
    )
    st.warning(
        "At least 28 consecutive daily observations are required for every series. "
        "All series use the saved recursive XGBoost model when its feature schema is "
        "compatible; product/store IDs do not need to be from M5. With fewer than 56 "
        "history days, the unavailable lag_56 feature is passed as missing to XGBoost. "
        "Prediction errors are shown instead of replaced with an average fallback. "
        "For dates beyond the M5 calendar, events and SNAP are set to zero and the "
        "last known price is carried forward; this can reduce accuracy."
    )
    uploaded_file = st.file_uploader(
        "Choose daily sales CSV",
        type=["csv"],
        key="sales_csv_upload",
    )

    if uploaded_file is not None:
        try:
            incoming = forecasting_service.validate_sales_rows(
                pd.read_csv(BytesIO(uploaded_file.getvalue()))
            )
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not validate the uploaded CSV: {exc}")
        else:
            st.caption(
                f"{len(incoming):,} uploaded rows · {incoming['id'].nunique():,} "
                f"series · latest sales date {incoming['date'].max():%B %d, %Y}"
            )
            st.dataframe(incoming.head(10), hide_index=True, width="stretch")
            latest_allowed_old_date = (
                pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            )
            is_historical_upload = incoming["date"].max() < latest_allowed_old_date
            if is_historical_upload:
                st.info(
                    f"ℹ️ **Historical Sales Detected**: The latest date in this upload is "
                    f"**{incoming['date'].max():%B %d, %Y}** (historical period). "
                    "You can generate a historical backtest forecast starting the day after this date, "
                    "or upload recent sales data for a genuine current operational forecast."
                )
            button_label = (
                "Generate 28-day historical forecast"
                if is_historical_upload
                else "Validate data and generate 28-day forecast"
            )
            if st.button(button_label):
                try:
                    bundle_path = active_model_bundle_path()
                    bundle = load_model_resource(
                        str(bundle_path),
                        bundle_path.stat().st_mtime_ns,
                    )
                    recorded_cutoff = str(
                        bundle["metadata"].get("training_end_day", "d_1941")
                    )
                    cutoff = (
                        recorded_cutoff
                        if re.fullmatch(r"d_[0-9]+", recorded_cutoff)
                        else "d_1941"
                    )
                    reference = load_m5_reference(
                        tuple(sorted(incoming["id"].astype(str).unique())),
                        cutoff,
                    )
                    combined_history = reference.get("history")
                    if forecasting_service.HISTORY_PATH.is_file():
                        saved_history = pd.read_parquet(
                            forecasting_service.HISTORY_PATH
                        )
                        combined_history = forecasting_service.merge_sales_history(
                            combined_history,
                            saved_history,
                        )
                    combined_history = forecasting_service.merge_sales_history(
                        combined_history,
                        incoming,
                    )
                    forecast_history = combined_history.loc[
                        combined_history["id"].isin(incoming["id"].unique())
                    ].reset_index(drop=True)
                    if CALENDAR_FILE.is_file():
                        calendar = pd.read_csv(CALENDAR_FILE)
                    else:
                        calendar = pd.DataFrame()
                    forecast, actual_history = forecasting_service.generate_forecast(
                        forecast_history,
                        bundle,
                        reference,
                        calendar,
                        horizon=28,
                        minimum_current_date=latest_allowed_old_date,
                        allow_historical=True,
                    )
                    details = forecasting_service.save_forecast_result(
                        forecast,
                        combined_history,
                        model_metadata=bundle["metadata"],
                        source_name=uploaded_file.name,
                    )
                except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
                    st.error(f"Could not generate the forecast: {exc}")
                else:
                    st.success(
                        f"{'Historical' if is_historical_upload else 'Current'} forecast saved for "
                        f"{details['forecast_start']} through {details['forecast_end']}."
                    )

    try:
        saved_result = load_saved_user_result()
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load the saved forecast: {exc}")
        saved_result = None

    if saved_result is None:
        st.info("Upload recent sales and generate a forecast to see results here.")
    else:
        forecast_all, actual_history_all, details = saved_result

        # Display comprehensive model metadata card
        with st.expander("ℹ️ Model Metadata & Provenance", expanded=True):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.markdown(f"**Model Version:** `{details.get('model_version', 'unknown')}`")
            meta_col1.markdown(f"**Algorithm:** {details.get('algorithm', 'XGBoost (count:poisson)')}")
            training_dt = details.get('training_date_utc')
            meta_col1.markdown(f"**Training Date (UTC):** {format_date(pd.Timestamp(training_dt)) if training_dt else 'Historical pre-trained'}")

            meta_col2.markdown(f"**Dataset:** {details.get('model_data_source') or 'M5 retail dataset'}")
            meta_col2.markdown(f"**Training Cutoff:** through {details.get('training_data_end_date') or details.get('training_end_day') or 'd_1941'}")
            d_ver = details.get('model_data_version')
            meta_col2.markdown(f"**Data Version:** `{str(d_ver)[:12]}`" if d_ver else "**Data Version:** `N/A`")

            eval_metrics = details.get("evaluation_metrics")
            if isinstance(eval_metrics, dict):
                meta_col3.markdown(f"**Backtest MAE:** `{eval_metrics.get('mean_mae', 'N/A')}` units")
                meta_col3.markdown(f"**Backtest RMSE:** `{eval_metrics.get('mean_rmse', 'N/A')}` units")
                meta_col3.markdown(f"**Backtest WAPE:** `{eval_metrics.get('mean_wape_pct', 'N/A')}%`")
            else:
                meta_col3.markdown("**Evaluation:** 3 rolling validation backtest periods available in Past accuracy")

        # Section: Overall Forecast Summary Across All Series
        st.subheader("📊 Overall Forecast Summary (All Uploaded Series)")
        total_pred_units = float(forecast_all["predicted_sales"].sum())
        num_series = forecast_all["id"].nunique()
        horizon_days = forecast_all["date"].nunique()

        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
        sum_c1.metric("Total Estimated Sales", format_units(total_pred_units))
        sum_c2.metric("Total Uploaded Series", f"{num_series:,}")
        sum_c3.metric("Forecast Horizon", f"{horizon_days} days")
        sum_c4.metric(
            "Forecast Window",
            f"{forecast_all['date'].min():%b %d} – {forecast_all['date'].max():%b %d, %Y}",
        )

        # Series breakdown table
        summary_table = (
            forecast_all.groupby("id", as_index=False)
            .agg(
                total_estimated_sales=("predicted_sales", "sum"),
                daily_average=("predicted_sales", "mean"),
                min_daily=("predicted_sales", "min"),
                max_daily=("predicted_sales", "max"),
                forecast_method=("forecast_method", "first"),
            )
            .sort_values("total_estimated_sales", ascending=False)
        )
        summary_table["Series Description"] = summary_table["id"].map(readable_series)
        display_summary = summary_table[
            [
                "id",
                "Series Description",
                "total_estimated_sales",
                "daily_average",
                "min_daily",
                "max_daily",
                "forecast_method",
            ]
        ].rename(
            columns={
                "id": "Series ID",
                "total_estimated_sales": "28-Day Total (units)",
                "daily_average": "Daily Avg (units)",
                "min_daily": "Min Day",
                "max_daily": "Max Day",
                "forecast_method": "Method",
            }
        )
        st.dataframe(display_summary.round(2), hide_index=True, width="stretch")

        st.divider()

        # Section: Individual Series Exploration
        st.subheader("🔍 Individual Series Forecast & History")
        series_options = ["All uploaded series"] + sorted(
            forecast_all["id"].astype(str).unique()
        )
        selected_series = st.selectbox(
            "Select Product / Store Series",
            series_options,
            format_func=lambda value: (
                value if value == "All uploaded series" else readable_series(value)
            ),
        )

        history_days = actual_history_all.loc[
            actual_history_all["date"]
            >= pd.Timestamp(details["as_of_date"]) - pd.Timedelta(days=89)
        ]
        if selected_series != "All uploaded series":
            history_days = history_days.loc[history_days["id"] == selected_series]
            forecast = forecast_all.loc[forecast_all["id"] == selected_series]
        else:
            history_days = history_days.loc[
                history_days["id"].isin(forecast_all["id"].unique())
            ]
            forecast = forecast_all

        # Historical Actuals Chart
        st.markdown(
            f"**Actual historical sales through {format_date(pd.Timestamp(details['as_of_date']))}**"
        )
        actual_chart = (
            history_days.groupby("date", as_index=True)["sales"]
            .sum()
            .rename("Actual sales")
            .to_frame()
        )
        st.line_chart(actual_chart)
        st.caption(
            "This chart contains recorded sales only. The predictions below are shown "
            "separately and do not get mixed into historical actuals."
        )

        # Forecast Predictions Chart
        is_hist_forecast = (
            details.get("forecast_type") == "historical"
            or forecast["date"].max() < pd.Timestamp.now().normalize()
        )
        forecast_title_prefix = "Historical backtest predictions" if is_hist_forecast else "Future predictions"
        st.markdown(
            f"**{forecast_title_prefix}: {format_date_range(forecast['date'].min(), forecast['date'].max())}**"
            + (" `(Historical Period)`" if is_hist_forecast else " `(Current Operational)`")
        )
        first, second, third = st.columns(3)
        first.metric(
            "Estimated sales over horizon",
            format_units(float(forecast["predicted_sales"].sum())),
        )
        second.metric(
            "Series included",
            f"{forecast['id'].nunique():,}",
        )
        third.metric(
            "Forecast methods",
            ", ".join(sorted(forecast["forecast_method"].unique())),
        )
        method_counts = forecast["forecast_method"].value_counts().to_dict()
        st.caption(
            "Prediction method for this selection: "
            + ", ".join(
                f"{method} ({count:,} daily predictions)"
                for method, count in sorted(method_counts.items())
            )
        )
        prediction_chart = (
            forecast.groupby("date", as_index=True)["predicted_sales"]
            .sum()
            .rename("Predicted sales")
            .to_frame()
        )
        st.line_chart(prediction_chart)
        st.caption(
            f"Saved forecast ID: {details['forecast_id']} · "
            f"Forecast data source: {details['data_source']}"
        )

elif page == "Past accuracy":
    try:
        results = load_backtest(str(BACKTEST_FILE))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load model results: {exc}")
        st.stop()

    st.subheader("How close were the sales estimates?")
    st.markdown(
        "We checked estimates against actual sales in three separate past 28-day periods. "
        "We also compared the model with a simple estimate: use each product's average "
        "sales from the previous 28 days."
    )
    st.info(
        "The model had a slightly smaller typical miss and percentage error in all three "
        "tests. The simple recent-average estimate had fewer very large misses and was "
        "closer to actual sales on average. Overall, neither method wins on every score."
    )

    display = results.rename(columns={
        "test_period": "Dates checked",
        "method": "Estimate method",
        "MAE": "Typical miss (units)",
        "RMSE": "Large-miss score (units)",
        "WAPE_pct": "Miss vs. sales (%)",
        "Bias": "Average over/under (units)",
    })
    st.dataframe(
        display[
            [
                "Dates checked",
                "Estimate method",
                "Typical miss (units)",
                "Large-miss score (units)",
                "Miss vs. sales (%)",
                "Average over/under (units)",
            ]
        ].round(4),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Typical prediction miss by date range")
    mae_chart = results.pivot(
        index="test_period",
        columns="method",
        values="MAE",
    )
    st.bar_chart(mae_chart)
    st.caption(
        "Shorter bars mean the estimates were closer to actual daily sales, on average."
    )

    if DIRECT_METRICS_FILE.is_file():
        direct_metrics = pd.read_csv(DIRECT_METRICS_FILE)
        overall = direct_metrics.loc[direct_metrics["horizon"] == 0]
        if not overall.empty:
            row = overall.iloc[0]
            st.subheader("Direct XGBoost candidate")
            st.caption(
                "This separate horizon-aware model was evaluated on the existing "
                "d_1914–d_1941 period, which is not an untouched final test."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Typical miss (units)": row["MAE"],
                            "Large-miss score (units)": row["RMSE"],
                            "Miss vs. sales (%)": row["WAPE_pct"],
                            "Average over/under (units)": row["Bias"],
                        }
                    ]
                ).round(4),
                width="stretch",
                hide_index=True,
            )

    try:
        historical_forecast = load_forecast(
            str(FORECAST_FILES["Past 28-day chart: estimates vs actual sales"][0]),
            has_actuals=True,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load detailed past-period results: {exc}")
    else:
        historical_forecast["error"] = (
            historical_forecast["predicted_sales"]
            - historical_forecast["actual_sales"]
        )
        with st.expander("Accuracy by product type or store"):
            segment = st.radio(
                "Group results by",
                ["Product type", "Store"],
                horizontal=True,
            )
            group_column = "category" if segment == "Product type" else "store"
            segment_rows = []
            for name, group in historical_forecast.groupby(group_column, sort=True):
                actual_total = float(group["actual_sales"].abs().sum())
                segment_rows.append(
                    {
                        segment: name,
                        "Typical miss (units)": group["error"].abs().mean(),
                        "Large-miss score (units)": (
                            group["error"].pow(2).mean() ** 0.5
                        ),
                        "Miss vs. sales (%)": (
                            100 * group["error"].abs().sum() / actual_total
                            if actual_total
                            else float("nan")
                        ),
                        "Average over/under (units)": group["error"].mean(),
                    }
                )
            st.dataframe(
                pd.DataFrame(segment_rows).round(4),
                width="stretch",
                hide_index=True,
            )

    direct_model = ROOT / "models" / "direct_xgboost.json"
    direct_metadata = ROOT / "models" / "direct_xgboost_metadata.json"
    if direct_model.is_file() and direct_metadata.is_file():
        st.subheader("Direct XGBoost feature importance")
        try:
            importance = load_direct_importance(
                str(direct_model),
                str(direct_metadata),
            )
        except (OSError, ValueError) as exc:
            st.error(f"Could not load feature importance: {exc}")
        else:
            st.bar_chart(importance.head(15).set_index("Feature"))

    with st.expander("How to read these scores"):
        st.markdown(
            "- **Typical miss:** the average size of the daily difference between the estimate "
            "and actual sales. Lower is better.\n"
            "- **Large-miss score:** also measures error, but gives extra weight to unusually "
            "large misses. Lower is better.\n"
            "- **Miss vs. sales:** the total size of the errors compared with total sales. "
            "Lower is better.\n"
            "- **Average over/under:** how far estimates were above or below actual sales on "
            "average. Close to zero is best."
        )

elif page == "Replenishment simulator":
    st.subheader("Replenishment scenario")
    st.warning(
        "This is a planning simulation, not an order recommendation. It uses "
        "assumptions you provide and requires human review."
    )
    try:
        saved_result = load_saved_user_result()
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load the saved forecast: {exc}")
        st.stop()
    if saved_result is None:
        st.info("Upload recent sales and create a forecast before running a simulation.")
        st.stop()

    forecast_raw, actual_history, details = saved_result
    forecast = forecast_raw.copy()

    # Parse metadata for all forecast series (handles M5, uploaded non-evaluation, and custom IDs)
    meta_df = pd.DataFrame(
        [
            {"id": sid, **parse_series_metadata(sid)}
            for sid in forecast["id"].unique()
        ]
    )
    forecast = forecast.merge(meta_df, on="id", how="left")

    if forecast.empty:
        st.error("No forecast series available for replenishment simulation.")
        st.stop()

    series = (
        forecast[["store", "item", "id", "category"]]
        .drop_duplicates()
        .sort_values(["store", "id"])
    )
    store_options = sorted(series["store"].unique())
    store = st.selectbox(
        "Store",
        store_options,
        format_func=readable_store,
        key="replenishment_store",
    )
    item_options = series.loc[series["store"] == store, "id"].tolist()
    selected_id = st.selectbox(
        "Product",
        item_options,
        format_func=readable_series,
        key="replenishment_item",
    )

    # Calculate historical metrics for guidance
    selected_hist = actual_history.loc[actual_history["id"] == selected_id].sort_values("date")
    hist_std = float(selected_hist["sales"].std()) if len(selected_hist) > 1 else 0.0
    hist_mean = float(selected_hist["sales"].mean()) if not selected_hist.empty else 0.0
    latest_price = None
    if "sell_price" in selected_hist.columns:
        valid_prices = selected_hist["sell_price"].dropna()
        if not valid_prices.empty:
            latest_price = float(valid_prices.iloc[-1])

    # Show item context card
    selected_fc = forecast.loc[forecast["id"] == selected_id].sort_values("date")
    daily_forecasts = selected_fc["predicted_sales"].astype(float).tolist()
    fc_sum_28 = sum(daily_forecasts)
    fc_mean_daily = fc_sum_28 / len(daily_forecasts) if daily_forecasts else 0.0

    st.caption(
        f"Selected series: `{selected_id}` · Historical mean sales: `{hist_mean:.1f}` units/day "
        f"(std: `{hist_std:.2f}`) · 28-day forecast: `{fc_sum_28:,.1f}` units "
        f"(avg: `{fc_mean_daily:.1f}` units/day)"
        + (f" · Latest unit price: `${latest_price:.2f}`" if latest_price is not None else "")
    )

    with st.form("replenishment_inputs"):
        st.markdown("##### 1. Inventory Status & Stock Positions")
        first, second, third = st.columns(3)
        on_hand = first.number_input(
            "Units currently on hand",
            min_value=0.0,
            value=float(round(hist_mean * 3, 0)),
            help="Usable stock physically available in the store/warehouse.",
        )
        on_order = second.number_input(
            "Units already on order",
            min_value=0.0,
            value=0.0,
            help="Open purchase orders or transfer shipments in transit.",
        )
        backorders = third.number_input(
            "Units backordered",
            min_value=0.0,
            value=0.0,
            help="Unfulfilled customer demand to be covered immediately.",
        )

        st.markdown("##### 2. Lead Time & Review Cadence")
        fourth, fifth, sixth = st.columns(3)
        lead_time = fourth.number_input(
            "Supplier lead time (days)",
            min_value=1,
            max_value=28,
            value=7,
            help="Days from order placement until stock arrives at store.",
        )
        review_period = fifth.number_input(
            "Review period (days)",
            min_value=0,
            max_value=27,
            value=0,
            help="Days between replenishment review cycles (0 for continuous review).",
        )
        pack_size = sixth.number_input(
            "Units per order pack / case",
            min_value=1,
            value=1,
            help="Orders are rounded up to whole pack multiples.",
        )

        st.markdown("##### 3. Service Level & Safety Stock Parameters")
        seventh, eighth = st.columns(2)
        minimum_order = seventh.number_input(
            "Minimum order quantity (MOQ)",
            min_value=0,
            value=0,
            help="Supplier minimum order constraint.",
        )
        service_level = eighth.selectbox(
            "Target cycle service level",
            [0.8, 0.9, 0.95, 0.99],
            index=2,
            format_func=lambda s: f"{int(s * 100)}% non-stockout probability",
        )
        daily_residual_std = st.number_input(
            "Daily forecast-error variability / safety stock factor (units)",
            min_value=0.0,
            value=float(round(hist_std, 2)),
            help=f"Pre-populated with historical std dev ({hist_std:.2f}). Set to 0 for zero safety stock.",
        )
        submitted = st.form_submit_button("Run simulation")

    if submitted:
        protection_days = int(lead_time + review_period)
        if protection_days > len(daily_forecasts):
            st.error(
                f"Protection period ({protection_days} days: {lead_time}d lead time + {review_period}d review) "
                f"exceeds the available forecast horizon ({len(daily_forecasts)} days)."
            )
        else:
            try:
                inputs = ReplenishmentInputs(
                    on_hand=on_hand,
                    on_order=on_order,
                    backorders=backorders,
                    lead_time_days=int(lead_time),
                    review_period_days=int(review_period),
                    pack_size=int(pack_size),
                    minimum_order_quantity=int(minimum_order),
                    service_level=service_level,
                    daily_residual_std=daily_residual_std,
                )
                result = simulate_replenishment(daily_forecasts, inputs)

                # If backend API is online, optionally log the run to SQLite
                if api_client.default_client.is_healthy():
                    try:
                        parsed_meta = parse_series_metadata(selected_id)
                        store_val = str(parsed_meta.get("store") or store)
                        item_val = str(parsed_meta.get("item") or selected_id)
                        api_payload = {
                            "store_id": store_val,
                            "item_id": item_val,
                            "forecast_type": "recursive",
                            "lead_time_days": int(lead_time),
                            "review_period_days": int(review_period),
                            "on_hand": float(on_hand),
                            "on_order": float(on_order),
                            "backorders": float(backorders),
                            "pack_size": int(pack_size),
                            "minimum_order_quantity": int(minimum_order),
                            "service_level": float(service_level),
                            "daily_residual_std": float(daily_residual_std),
                        }
                        api_client.default_client.simulate_replenishment(api_payload)
                    except Exception:
                        pass
            except ValueError as exc:
                st.error(f"Could not run simulation: {exc}")
            else:
                st.markdown("#### Simulation Results")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric(
                    "Current Inventory Position",
                    f"{result['inventory_position']:,.1f} units",
                    help="On-hand + On-order - Backorders",
                )
                m_c2.metric(
                    "Protection Period Demand",
                    f"{result['protection_period_demand']:,.1f} units",
                    help=f"Expected demand over {protection_days} days",
                )
                m_c3.metric(
                    "Safety Stock Required",
                    f"{result['protection_period_safety_stock']:,.1f} units",
                    help=f"Safety buffer for {int(service_level*100)}% service level",
                )
                m_c4.metric(
                    "Reorder Point",
                    f"{result['reorder_point']:,.1f} units",
                    help="Threshold below which an order is triggered",
                )

                suggested_qty = int(result["suggested_order_quantity"])
                st.subheader(
                    f"Suggested Replenishment Order: **{suggested_qty:,} units**"
                    + (f" ({suggested_qty // int(pack_size):,} packs)" if int(pack_size) > 1 else "")
                )
                if latest_price is not None and suggested_qty > 0:
                    est_cost = suggested_qty * latest_price
                    st.caption(f"Estimated order cost at ${latest_price:.2f}/unit: **${est_cost:,.2f}**")

                if result["inventory_position"] <= result["reorder_point"]:
                    st.warning(f"⚠️ {result['severity']}")
                else:
                    st.success(f"✅ {result['severity']}")
                st.info(f"ℹ️ {result['disclaimer']}")

elif page == "Data quality":
    st.subheader("Dataset checks")
    report = None
    if api_client.default_client.is_healthy():
        try:
            report = api_client.default_client.get_data_quality()
        except Exception:
            report = None
    if report is None:
        try:
            report = load_quality_report(str(QUALITY_REPORT))
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.info(str(exc))
            st.code("python .\\data_quality.py", language="powershell")
            st.stop()
    st.caption(
        f"Data version: {report['data_version']} · "
        f"Report created: {report['generated_at_utc']}"
    )
    sales_quality = report["sales"]
    calendar_quality = report["calendar"]
    price_quality = report["prices"]
    first, second, third = st.columns(3)
    first.metric("Product/store series", f"{sales_quality['series_count']:,}")
    second.metric("Sales days", f"{sales_quality['day_count']:,}")
    third.metric("Price rows checked", f"{price_quality['row_count']:,}")
    st.json(
        {
            "Sales": sales_quality,
            "Calendar": calendar_quality,
            "Prices": price_quality,
            "Limitations": report["limitations"],
        },
        expanded=False,
    )

elif page == "Train and save M5 forecast":
    st.subheader("Optional manual model retraining")
    st.markdown(
        "Forecast requests and CSV uploads never train the model. Choose one of the "
        "manual training options below only when you intentionally want to replace "
        "the active saved XGBoost model."
    )
    st.warning(
        "Training a new model replaces the model used for future forecasts. Keep a "
        "copy of the current model if you need to roll back."
    )
    st.caption(
        "A model is loaded and reused for predictions. No retraining starts when you "
        "open the app or upload sales."
    )

    st.markdown("#### Retrain from saved uploaded sales")
    uploaded_training_available = forecasting_service.HISTORY_PATH.is_file()
    current_job = training_jobs.read_latest_job()
    job_running = bool(
        current_job
        and current_job.get("status") in {"starting", "running"}
    )
    if uploaded_training_available:
        saved_history = pd.read_parquet(forecasting_service.HISTORY_PATH)
        max_days_available = int(saved_history.groupby("id")["date"].nunique().max())
        min_days_available = int(saved_history.groupby("id")["date"].nunique().min())
        st.caption(
            f"Saved history: {len(saved_history):,} rows across "
            f"{saved_history['id'].nunique():,} series, through "
            f"{pd.to_datetime(saved_history['date']).max():%B %d, %Y}. "
            f"Observed history spans {min_days_available}–{max_days_available} consecutive days per series."
        )

        with st.expander("⚙️ Training Requirements & Hyperparameters", expanded=False):
            cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
            min_days_input = cfg_c1.number_input(
                "Minimum consecutive history days",
                min_value=29,
                max_value=365,
                value=min(max_days_available, 29),
                help="Needs at least 29 days (28 history days for lag_28/rolling features + 1 target day).",
            )
            min_rows_input = cfg_c2.number_input(
                "Minimum training examples required",
                min_value=5,
                max_value=10_000,
                value=10,
                help="Total training instances across all eligible series.",
            )
            boost_rounds_input = cfg_c3.number_input(
                "Boosting rounds",
                min_value=10,
                max_value=500,
                value=50,
                help="Number of gradient boosted decision trees.",
            )
    else:
        st.info("Generate a forecast from recent sales first to save upload history.")
        min_days_input = 29
        min_rows_input = 10
        boost_rounds_input = 50

    if "uploaded_retrain_success" in st.session_state:
        st.success(st.session_state.pop("uploaded_retrain_success"))

    confirm_uploaded_training = st.checkbox(
        "I understand this will train and activate a new model from saved uploaded sales."
    )
    if job_running:
        st.warning("⏳ An M5 training job is currently running in the background. Retraining is disabled until it finishes or is cancelled.")
        if st.button("⏹️ Cancel active training job", key="cancel_job_retrain_top"):
            training_jobs.cancel_active_job()
            st.rerun()

    start_uploaded_training = st.button(
        "Train from saved uploaded sales",
        disabled=(
            not uploaded_training_available
            or not confirm_uploaded_training
            or job_running
        ),
    )
    if start_uploaded_training:
        try:
            saved_history = pd.read_parquet(forecasting_service.HISTORY_PATH)
            calendar = (
                pd.read_csv(CALENDAR_FILE)
                if CALENDAR_FILE.is_file()
                else pd.DataFrame()
            )
            reference = load_m5_reference(
                tuple(sorted(saved_history["id"].astype(str).unique())),
                "d_1941",
            )
            with st.spinner("Building training features and fitting XGBoost..."):
                metadata = forecasting_service.train_uploaded_history(
                    saved_history,
                    calendar,
                    reference,
                    minimum_days=int(min_days_input),
                    minimum_rows=int(min_rows_input),
                    num_boost_round=int(boost_rounds_input),
                )
        except Exception as exc:
            st.error(f"Could not retrain from uploaded sales: {exc}")
        else:
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state["uploaded_retrain_success"] = (
                f"Model `{metadata['model_version']}` trained on "
                f"{metadata['training_rows']:,} examples from "
                f"{metadata['eligible_series']:,} eligible series and activated."
            )
            st.rerun()

    st.markdown("#### Retrain on the full M5 training dataset")
    st.markdown(
        "This separate long-running background workflow trains from the existing M5 feature "
        "table and saves a versioned `.joblib` model bundle."
    )
    required_training_files = [
        training_jobs.FEATURE_FILE,
        training_jobs.SALES_FILE,
        training_jobs.CALENDAR_FILE,
        training_jobs.PRICES_FILE,
    ]
    missing_training_files = [
        str(path) for path in required_training_files if not path.is_file()
    ]
    if missing_training_files:
        st.error(
            "Manual M5 retraining is unavailable until these local files are prepared: "
            + ", ".join(missing_training_files)
        )

    with st.expander("⚙️ M5 Retraining Settings", expanded=False):
        m5_rounds_input = st.number_input(
            "M5 Boosting Rounds",
            min_value=10,
            max_value=300,
            value=150,
            step=25,
            help="Configures the number of boosting trees for M5 full-dataset training.",
        )

    confirm_long_run = st.checkbox(
        "I understand this starts a long model-training job on the M5 data."
    )
    start_training = st.button(
        "Start training and save forecast",
        disabled=(
            not confirm_long_run
            or bool(missing_training_files)
            or job_running
        ),
    )

    if start_training:
        try:
            new_job = training_jobs.start_recursive_training(num_boost_round=int(m5_rounds_input))
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(f"Could not start training: {exc}")
        else:
            st.success(f"Training job {new_job['job_id']} started.")
            st.rerun()

    _training_status_panel()

elif page == "About this tool":
    st.subheader("About this sales-estimate tool")
    st.markdown(
        """
        This tool demonstrates how a computer model can estimate daily sales for
        different products at different stores, using the M5 retail dataset.

        **What the dashboard shows**
        - Uploaded historical actual sales, shown separately from future predictions.
        - A 28-day forecast generated from recent uploaded sales with the saved model.
        - Model version, available training-date provenance, and forecast data source.

        **Important limitation**

        The saved recursive XGBoost model uses numeric calendar, price, lag, and
        rolling-sales features and can score product/store IDs that were not in M5.
        It requires 28 consecutive observations; with fewer than 56 days, the missing
        lag_56 is passed to XGBoost as missing. Prediction errors are shown rather than
        replaced with a constant average. For dates beyond the M5 calendar, event and
        SNAP features are zero and the latest available price is carried forward.
        This project is not validated for every retailer or future market condition.

        XGBoost was retained because the saved recursive comparison reported a lower
        recursive MAE than the tested LightGBM and two-stage candidates.

        **How to use it**

        Choose a date range, then select a product type and store. Choose one product
        to see its daily sales estimate, or select all products to see the combined
        demand for your filters.
        """
    )
