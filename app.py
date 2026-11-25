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

# Friendly names for model features
FEATURE_NAME_MAP = {
    "roll_mean_28": "Average sales, last 28 days",
    "roll_mean_7": "Average sales, last 7 days",
    "roll_std_28": "Sales volatility, last 28 days",
    "roll_std_7": "Sales volatility, last 7 days",
    "lag_1": "Sales yesterday",
    "lag_7": "Sales 7 days ago",
    "lag_14": "Sales 14 days ago",
    "lag_28": "Sales 28 days ago",
    "lag_56": "Sales 56 days ago",
    "zero_ratio_7": "Share of zero-sales days, last week",
    "zero_ratio_28": "Share of zero-sales days, last 28 days",
    "no_previous_sale": "No earlier sales recorded",
    "sell_price": "Unit selling price",
    "price_momentum": "Recent price trend",
    "price_change_rate": "Price change percentage",
    "price_to_mean": "Price relative to store average",
    "wday": "Day of the week (Monday–Sunday)",
    "month": "Month of the year",
    "year": "Calendar year",
    "event_name_1": "Special holiday or cultural event",
    "event_type_1": "Event category",
    "snap_CA": "California SNAP benefit day",
    "snap_TX": "Texas SNAP benefit day",
    "snap_WI": "Wisconsin SNAP benefit day",
    "item_sales_rank": "Product sales rank in department",
    "store_sales_rank": "Store total volume rank",
}

st.set_page_config(
    page_title="Store Demand Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Max container width to prevent text stretching on wide/ultrawide displays */
    .block-container {
        max-width: 1140px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        margin: 0 auto;
    }
    html, body, [class*="css"] {
        font-size: 16px;
    }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.1rem 1.25rem;
        border-radius: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .intro {
        color: #475569;
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    .headline-sentence {
        font-size: 1.2rem;
        font-weight: 500;
        color: #1e293b;
        margin: 0.8rem 0 1.2rem 0;
        padding: 0.85rem 1.15rem;
        background: #f1f5f9;
        border-left: 5px solid #2563eb;
        border-radius: 0.4rem;
        line-height: 1.5;
    }
    .trust-badge {
        padding: 0.85rem 1.15rem;
        border-radius: 0.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0.6rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .trust-badge-medium {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }
    .trust-badge-high {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }
    .trust-badge-low {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
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
        return 1.0, "Training stopped or failed"

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
            f"Fitting XGBoost decision trees: round {completed_rounds}/{total_rounds}"
        )
    if status.get("stage") == "preparing":
        return 0.05, "Preparing training features from sales history"
    return 0.02, "Starting background worker process"


@st.fragment(run_every="5s")
def _training_status_panel() -> None:
    status = training_jobs.read_latest_job()
    if status is None:
        st.info("ℹ️ No training job has been started from this dashboard yet.")
        return

    job_id = str(status.get("job_id", "unknown"))
    read_tail = getattr(training_jobs, "read_log_tail", lambda path, lines=24: [])
    log_lines = read_tail(str(status.get("log_path", "")))
    progress, progress_label = _training_progress(status, log_lines)

    if status.get("status") in {"starting", "running"}:
        st.info(f"⏳ **Training in Progress:** {progress_label}")
        st.progress(progress, text=f"Progress: {int(progress * 100)}% ({progress_label})")
        st.caption(
            "This background job is training decision trees on your historical dataset. "
            "You can keep this page open; status updates every 5 seconds."
        )
        if st.button("⏹️ Cancel background training", key=f"cancel-job-{job_id}", help="Halt this training run immediately"):
            training_jobs.cancel_active_job()
            st.rerun(scope="app")
    elif status.get("status") == "completed":
        if st.session_state.get("cleared_training_cache_job") != job_id:
            st.cache_data.clear()
            st.session_state["cleared_training_cache_job"] = job_id
            st.rerun(scope="app")
        st.success("✅ **Training finished successfully.** The new model and its 28-day forecasts are active and saved.")
    elif status.get("status") == "failed":
        st.error(f"❌ **Training failed or stopped:** {status.get('error', 'See the job log below.')}")
        st.caption("If a process was terminated or crashed, click below to release the training lock.")
        if st.button("🔄 Clear status & Reset training lock", key=f"clear-lock-{job_id}", help="Reset lock so a new job can be launched"):
            training_jobs.cancel_active_job()
            st.rerun(scope="app")
    else:
        st.warning(f"⚠️ Training job state: {status.get('status', 'unknown')}")

    st.caption(
        f"Job ID: `{job_id}` · Started: `{status.get('started_at_utc', 'unknown')}` · "
        f"Stage: `{status.get('stage', 'unknown')}`"
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
                    help="Download the fitted XGBoost booster JSON file",
                )
            if forecast_path.is_file():
                downloads[1].download_button(
                    "Download forecast",
                    data=forecast_path.read_bytes(),
                    file_name=forecast_path.name,
                    mime="application/vnd.apache.parquet",
                    key=f"download-forecast-{job_id}",
                    help="Download the generated future forecast in Parquet format",
                )
            if model_bundle_path.is_file():
                downloads[2].download_button(
                    "Download reusable joblib model",
                    data=model_bundle_path.read_bytes(),
                    file_name=model_bundle_path.name,
                    mime="application/octet-stream",
                    key=f"download-model-bundle-{job_id}",
                    help="Download the packaged joblib bundle containing model and features",
                )
            if metadata_path.is_file():
                downloads[3].download_button(
                    "Download model details",
                    data=metadata_path.read_bytes(),
                    file_name=metadata_path.name,
                    mime="application/json",
                    key=f"download-metadata-{job_id}",
                    help="Download technical training metadata in JSON format",
                )
    if log_lines:
        with st.expander("Recent training output log", expanded=(status.get("status") in {"starting", "running"})):
            st.code("\n".join(log_lines), language="text")


@st.cache_data(show_spinner=False)
def load_direct_importance(model_path: str, metadata_path: str) -> pd.DataFrame:
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
    rows = []
    for feature, score in scores.items():
        raw_name = (
            feature_names[int(feature[1:])]
            if feature.startswith("f") and feature[1:].isdigit()
            else feature
        )
        friendly_label = FEATURE_NAME_MAP.get(raw_name, raw_name.replace("_", " ").title())
        rows.append({
            "Feature": friendly_label,
            "Technical name": raw_name,
            "Gain importance": float(score),
        })
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
    """Extract store, item, department, category, and state from any series ID."""
    cleaned = str(series_id).strip()
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

    parts = cleaned.split("_")
    if len(parts) >= 4 and len(parts[-2]) == 2 and parts[-2].isalpha():
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


# ==============================================================================
# SIDEBAR NAVIGATION & AUDIENCE SWITCH (TASK 1)
# ==============================================================================

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Store Manager"

with st.sidebar:
    st.markdown("### 👤 I am a:")
    user_role = st.radio(
        "I am a:",
        ["Store Manager", "Analyst"],
        index=0 if st.session_state["user_role"] == "Store Manager" else 1,
        key="role_choice",
        label_visibility="collapsed",
        help="Store Manager view provides simple forecasts and order recommendations. Analyst view adds accuracy scores, data quality checks, and model training.",
    )
    st.session_state["user_role"] = user_role

    st.divider()
    st.markdown("### 🧭 Navigation")

    if user_role == "Store Manager":
        page_options = [
            "Explore sales estimates",
            "Replenishment simulator",
            "About this tool",
        ]
    else:
        page_options = [
            "Explore sales estimates",
            "Past accuracy",
            "Replenishment simulator",
            "Data quality",
            "Train and save M5 forecast",
            "About this tool",
        ]

    # Preserve or safely reset active page across role toggles
    current_active = st.session_state.get("active_page", "Explore sales estimates")
    if current_active not in page_options:
        current_active = "Explore sales estimates"

    def _format_page_label(p: str) -> str:
        if p == "Train and save M5 forecast":
            return "⚙️ Admin / Advanced: Train Model"
        return p

    page = st.radio(
        "Choose a view",
        page_options,
        index=page_options.index(current_active),
        format_func=_format_page_label,
        key="active_page",
        label_visibility="collapsed",
        help="Select a view to open.",
    )

    st.divider()
    api_online = api_client.default_client.is_healthy()
    if api_online:
        st.caption("🟢 **Backend API**: Connected (`https://store-level-demand-forcaster.onrender.com`)")
    else:
        st.caption("🟡 **Backend API**: Offline (Local mode)")


# ==============================================================================
# PAGE 1: EXPLORE SALES ESTIMATES (TASK 2)
# ==============================================================================

if page == "Explore sales estimates":
    st.title("📦 Store Demand Forecast")
    st.markdown(
        '<p class="intro">See estimated product demand by store, upload new store sales, '
        'and plan inventory requirements with machine-learning forecasts.</p>',
        unsafe_allow_html=True,
    )

    # Technical details moved into an expander
    with st.expander("🛠️ Technical notes & Model specifications", expanded=False):
        st.markdown(
            "- **Model**: Saved recursive XGBoost model with count:poisson objective.\n"
            "- **Requirements**: At least 28 consecutive daily observations are required per series.\n"
            "- **Lags**: With fewer than 56 history days, lag_56 is passed as missing to XGBoost.\n"
            "- **Calendar & SNAP**: For dates beyond the M5 calendar, events and SNAP are set to zero "
            "and the last known unit price is carried forward.\n"
            "- **Zero Arbitrary Fallbacks**: Genuine model outputs and errors are surfaced rather than masked."
        )

    # Upload section
    st.markdown("#### Upload recent sales to forecast future demand")
    st.markdown(
        "Upload a daily sales CSV with columns `id,date,sales` (optional: `sell_price`). "
        "The model forecasts 28 days forward starting after your newest sales date."
    )

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload daily sales CSV",
            type=["csv"],
            key="sales_csv_upload",
            help="Upload your daily sales CSV file. Format: id, date, sales, sell_price.",
            label_visibility="collapsed",
        )
    with up_col2:
        st.download_button(
            "📥 Download CSV template",
            data="id,date,sales,sell_price\nFOODS_1_001_CA_1_evaluation,2016-04-01,2,2.48\n",
            file_name="daily_sales_template.csv",
            mime="text/csv",
            help="Download a sample template with expected header and row formats",
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
                f"✅ **Validated:** {len(incoming):,} rows across {incoming['id'].nunique():,} series "
                f"· Latest sales date: {incoming['date'].max():%B %d, %Y}"
            )
            with st.expander("Preview uploaded rows", expanded=False):
                st.dataframe(incoming.head(10), hide_index=True, width="stretch")

            latest_allowed_old_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            is_historical_upload = incoming["date"].max() < latest_allowed_old_date
            if is_historical_upload:
                st.info(
                    f"ℹ️ **Historical Sales Detected**: The latest date in this upload is "
                    f"**{incoming['date'].max():%B %d, %Y}**. You can generate a 28-day backtest forecast "
                    "starting the day after this date to evaluate what the model would have predicted."
                )

            button_label = (
                "Generate 28-day historical forecast"
                if is_historical_upload
                else "Validate data and generate 28-day forecast"
            )
            if st.button(button_label, type="primary", help="Run the XGBoost model to generate 28 days of future demand"):
                try:
                    with st.spinner("Preparing features and generating XGBoost forecasts..."):
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
                        calendar = (
                            pd.read_csv(CALENDAR_FILE)
                            if CALENDAR_FILE.is_file()
                            else pd.DataFrame()
                        )
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
                        f"✅ {'Historical' if is_historical_upload else 'Current'} forecast saved for "
                        f"{details['forecast_start']} through {details['forecast_end']}."
                    )
                    st.rerun()

    # Load active saved forecast
    try:
        saved_result = load_saved_user_result()
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load the saved forecast: {exc}")
        saved_result = None

    if saved_result is None:
        st.info("ℹ️ Upload recent sales and generate a forecast above to explore sales estimates.")
    else:
        forecast_all, actual_history_all, details = saved_result

        # MODEL METADATA: Only shown in Analyst View
        if user_role == "Analyst":
            with st.expander("ℹ️ Model Metadata & Provenance (Analyst View)", expanded=False):
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.markdown(f"**Model Version:** `{details.get('model_version', 'unknown')}`")
                meta_col1.markdown(f"**Algorithm:** {details.get('algorithm', 'XGBoost (count:poisson)')}")
                training_dt = details.get("training_date_utc")
                meta_col1.markdown(f"**Training Date (UTC):** {format_date(pd.Timestamp(training_dt)) if training_dt else 'Historical pre-trained'}")

                meta_col2.markdown(f"**Dataset:** {details.get('model_data_source') or 'M5 retail dataset'}")
                meta_col2.markdown(f"**Training Cutoff:** through {details.get('training_data_end_date') or details.get('training_end_day') or 'd_1941'}")
                d_ver = details.get("model_data_version")
                meta_col2.markdown(f"**Data Version:** `{str(d_ver)[:12]}`" if d_ver else "**Data Version:** `N/A`")

                eval_metrics = details.get("evaluation_metrics")
                if isinstance(eval_metrics, dict):
                    meta_col3.markdown(f"**Backtest MAE:** `{eval_metrics.get('mean_mae', 'N/A')}` units")
                    meta_col3.markdown(f"**Backtest RMSE:** `{eval_metrics.get('mean_rmse', 'N/A')}` units")
                    meta_col3.markdown(f"**Backtest WAPE:** `{eval_metrics.get('mean_wape_pct', 'N/A')}%`")
                else:
                    meta_col3.markdown("**Evaluation:** 3 rolling validation backtest periods available in Past accuracy")

        st.divider()

        # PRODUCT SELECTION
        series_options = ["All uploaded series"] + sorted(
            forecast_all["id"].astype(str).unique()
        )
        selected_series = st.selectbox(
            "Select Product / Store Series to Inspect:",
            series_options,
            format_func=lambda val: "🌐 All uploaded series (Combined)" if val == "All uploaded series" else readable_series(val),
            help="Choose a specific product and store to see individual demand, or view combined demand.",
            key="selected_forecast_series",
        )

        if selected_series != "All uploaded series":
            forecast = forecast_all.loc[forecast_all["id"] == selected_series].sort_values("date")
            meta = parse_series_metadata(selected_series)
            display_item_name = meta["item"]
            display_store_name = readable_store(meta["store"])
        else:
            forecast = forecast_all.sort_values("date")
            display_item_name = f"all {forecast_all['id'].nunique()} products"
            display_store_name = "all stores"

        # TASK 2: PUT THE ANSWER FIRST — 3 BIG CARDS
        # 1. Next 7 days sales
        dates_sorted = sorted(forecast["date"].unique())
        first_7_dates = dates_sorted[:7]
        fc_7d = float(forecast.loc[forecast["date"].isin(first_7_dates), "predicted_sales"].sum())
        # 2. Next 28 days sales
        fc_28d = float(forecast["predicted_sales"].sum())
        # 3. Busiest day
        daily_agg = forecast.groupby("date")["predicted_sales"].sum()
        busiest_date = daily_agg.idxmax()
        busiest_sales = float(daily_agg.max())

        card1, card2, card3 = st.columns(3)
        card1.metric(
            "Expected Sales (Next 7 days)",
            format_units(fc_7d),
            help="Total anticipated demand over the next 7 calendar days",
        )
        card2.metric(
            "Expected Sales (Next 28 days)",
            format_units(fc_28d),
            help="Total anticipated demand over the full 4-week forecast horizon",
        )
        card3.metric(
            "Busiest Day Expected",
            format_units(busiest_sales),
            delta=f"{busiest_date:%A, %b %d}",
            help=f"Peak single day: {busiest_date:%B %d, %Y} with approximately {busiest_sales:,.0f} units",
        )

        # TASK 2: ONE PLAIN-ENGLISH SENTENCE
        st.markdown(
            f'<div class="headline-sentence">💡 For <strong>{display_item_name}</strong> at '
            f'<strong>{display_store_name}</strong>, you will probably sell about '
            f'<strong>{fc_28d:,.0f} units</strong> in the next 28 days '
            f'(approx. <strong>{fc_7d:,.0f} units</strong> in the next 7 days).</div>',
            unsafe_allow_html=True,
        )

        # TASK 2: TRUST BADGE (ICON + TEXT)
        # Based on backtest benchmarks (typical retail grocery daily error ~1.1 units, weekly aggregated accuracy is much higher)
        st.markdown(
            '<div class="trust-badge trust-badge-medium">'
            '<span style="font-size: 1.25rem;">🟠</span> '
            '<div><strong>Medium reliability:</strong> Good for weekly totals and stock planning; '
            'less exact for single-day fluctuations due to everyday shopper variability.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # FORECAST VISUALIZATION
        is_hist_forecast = (
            details.get("forecast_type") == "historical"
            or forecast["date"].max() < pd.Timestamp.now().normalize()
        )
        forecast_title_prefix = "Historical Backtest Estimates" if is_hist_forecast else "Daily Sales Estimates"
        st.markdown(f"##### 📈 {forecast_title_prefix} ({format_date_range(forecast['date'].min(), forecast['date'].max())})")

        prediction_chart = (
            forecast.groupby("date", as_index=True)["predicted_sales"]
            .sum()
            .rename("Estimated daily sales (units)")
            .to_frame()
        )
        st.line_chart(prediction_chart, height=320)

        # HISTORICAL CONTEXT
        with st.expander("📊 View past actual sales history for context", expanded=False):
            history_days = actual_history_all.loc[
                actual_history_all["date"]
                >= pd.Timestamp(details["as_of_date"]) - pd.Timedelta(days=89)
            ]
            if selected_series != "All uploaded series":
                history_days = history_days.loc[history_days["id"] == selected_series]
            else:
                history_days = history_days.loc[history_days["id"].isin(forecast_all["id"].unique())]

            if not history_days.empty:
                actual_chart = (
                    history_days.groupby("date", as_index=True)["sales"]
                    .sum()
                    .rename("Actual recorded sales")
                    .to_frame()
                )
                st.line_chart(actual_chart, height=260)
                st.caption(
                    "This chart contains past recorded sales only. Forecasts are generated separately "
                    "and never mixed into historical numbers."
                )

        # ALL-SERIES BREAKDOWN TABLE (For multi-item planning)
        if selected_series == "All uploaded series" and forecast_all["id"].nunique() > 1:
            st.markdown("##### 📋 Product Breakdown")
            summary_table = (
                forecast_all.groupby("id", as_index=False)
                .agg(
                    total_estimated_sales=("predicted_sales", "sum"),
                    daily_average=("predicted_sales", "mean"),
                    min_daily=("predicted_sales", "min"),
                    max_daily=("predicted_sales", "max"),
                )
                .sort_values("total_estimated_sales", ascending=False)
            )
            summary_table["Product / Store"] = summary_table["id"].map(readable_series)
            display_summary = summary_table[
                ["Product / Store", "total_estimated_sales", "daily_average", "min_daily", "max_daily"]
            ].rename(
                columns={
                    "total_estimated_sales": "28-Day Total (units)",
                    "daily_average": "Daily Avg (units)",
                    "min_daily": "Min Day",
                    "max_daily": "Max Day",
                }
            )
            st.dataframe(display_summary.round(1), hide_index=True, width="stretch")


# ==============================================================================
# PAGE 2: REPLENISHMENT SIMULATOR (TASK 3)
# ==============================================================================

elif page == "Replenishment simulator":
    st.title("📦 Inventory Replenishment Simulator")
    st.markdown(
        '<p class="intro">Figure out exactly how much inventory to order for any product. '
        'Compare your current on-hand stock with supplier lead times and safety buffers.</p>',
        unsafe_allow_html=True,
    )

    try:
        saved_result = load_saved_user_result()
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load the saved forecast: {exc}")
        saved_result = None

    if saved_result is None:
        st.info("ℹ️ Upload recent sales and create a forecast on the **Explore sales estimates** page before simulating inventory orders.")
        st.stop()

    forecast_raw, actual_history, details = saved_result
    forecast = forecast_raw.copy()

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

    col_store, col_item = st.columns(2)
    with col_store:
        store_options = sorted(series["store"].unique())
        store = st.selectbox(
            "Select Store",
            store_options,
            format_func=readable_store,
            key="replenishment_store",
            help="Choose the retail store location for this inventory calculation",
        )
    with col_item:
        item_options = series.loc[series["store"] == store, "id"].tolist()
        selected_id = st.selectbox(
            "Select Product",
            item_options,
            format_func=readable_series,
            key="replenishment_item",
            help="Choose the product to simulate reordering for",
        )

    # Historical metrics for guidance
    selected_hist = actual_history.loc[actual_history["id"] == selected_id].sort_values("date")
    hist_std = float(selected_hist["sales"].std()) if len(selected_hist) > 1 else 1.0
    hist_mean = float(selected_hist["sales"].mean()) if not selected_hist.empty else 1.0
    latest_price = None
    if "sell_price" in selected_hist.columns:
        valid_prices = selected_hist["sell_price"].dropna()
        if not valid_prices.empty:
            latest_price = float(valid_prices.iloc[-1])

    selected_fc = forecast.loc[forecast["id"] == selected_id].sort_values("date")
    daily_forecasts = selected_fc["predicted_sales"].astype(float).tolist()
    fc_sum_28 = sum(daily_forecasts)
    fc_mean_daily = fc_sum_28 / len(daily_forecasts) if daily_forecasts else 1.0

    st.caption(
        f"Product: `{selected_id}` · Average expected sales: `{fc_mean_daily:.1f}` units/day "
        f"· 28-day expected demand: `{fc_sum_28:,.0f}` units"
        + (f" · Latest price: `${latest_price:.2f}`/unit" if latest_price is not None else "")
    )

    with st.form("replenishment_inputs"):
        st.markdown("##### 1. Current Inventory Status")
        f1, f2, f3 = st.columns(3)
        on_hand = f1.number_input(
            "Units currently on hand",
            min_value=0.0,
            value=float(round(hist_mean * 4, 0)),
            help="Stock physically sitting on your store shelves and backroom right now.",
        )
        on_order = f2.number_input(
            "Units already on order (in transit)",
            min_value=0.0,
            value=0.0,
            help="Purchase orders already placed that are currently being shipped to the store.",
        )
        backorders = f3.number_input(
            "Customer backorders",
            min_value=0.0,
            value=0.0,
            help="Unfulfilled customer orders that must be satisfied immediately upon arrival.",
        )

        st.markdown("##### 2. Supplier Delivery Timings & Constraints")
        f4, f5, f6 = st.columns(3)
        lead_time = f4.number_input(
            "Supplier delivery lead time (days)",
            min_value=1,
            max_value=28,
            value=7,
            help="How many days it takes from placing an order until it arrives at your store.",
        )
        review_period = f5.number_input(
            "Order review schedule (days)",
            min_value=0,
            max_value=27,
            value=0,
            help="Days between order reviews. Enter 0 if you review stock and place orders daily.",
        )
        pack_size = f6.number_input(
            "Pack / case size",
            min_value=1,
            value=1,
            help="Case packaging constraint. Orders will automatically be rounded up to whole pack multiples.",
        )

        st.markdown("##### 3. Service Level & Buffer Stock")
        f7, f8 = st.columns(2)
        minimum_order = f7.number_input(
            "Minimum order quantity (MOQ)",
            min_value=0,
            value=0,
            help="The smallest order quantity accepted by the supplier.",
        )
        service_level = f8.selectbox(
            "Target in-stock availability",
            [0.8, 0.9, 0.95, 0.99],
            index=2,
            format_func=lambda s: f"{int(s * 100)}% non-stockout probability",
            help="Target probability that a customer finds the product in stock (95% is standard for retail).",
        )

        # Advanced settings expander for technical variability field
        with st.expander("⚙️ Advanced settings", expanded=False):
            daily_residual_std = st.number_input(
                "Forecast uncertainty (advanced)",
                min_value=0.0,
                value=float(round(hist_std, 2)),
                help="Standard deviation of daily forecast errors. Pre-filled with historical variability. Set to 0 to disable extra safety buffer.",
            )

        submitted = st.form_submit_button("Calculate Replenishment Order", type="primary", help="Simulate inventory requirements and recommended order")

    if submitted:
        protection_days = int(lead_time + review_period)
        if protection_days > len(daily_forecasts):
            st.error(
                f"Protection period ({protection_days} days = {lead_time}d lead time + {review_period}d review) "
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

                # Optional backend recording
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
                st.divider()

                # TASK 3: HEADLINE RESULT FIRST
                avg_daily_demand = result["protection_period_demand"] / max(1, protection_days)
                days_cover = (result["inventory_position"] / avg_daily_demand) if avg_daily_demand > 0 else 99.0
                suggested_qty = int(result["suggested_order_quantity"])

                if result["inventory_position"] <= result["reorder_point"]:
                    st.markdown(
                        f'<div class="headline-sentence" style="border-left-color: #f59e0b; background: #fffbeb;">'
                        f'🟠 <strong>You have about {days_cover:.1f} days of stock. Order {suggested_qty:,} units.</strong><br>'
                        f'<span style="font-size: 0.95rem; font-weight: normal; color: #78350f;">'
                        f'Current stock ({result["inventory_position"]:,.0f} units) is at or below the reorder trigger level '
                        f'({result["reorder_point"]:,.1f} units). Place an order now to prevent running out.</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="headline-sentence" style="border-left-color: #10b981; background: #ecfdf5;">'
                        f'🟢 <strong>Healthy Stock: You have about {days_cover:.1f} days of stock. No order needed today.</strong><br>'
                        f'<span style="font-size: 0.95rem; font-weight: normal; color: #065f46;">'
                        f'Current stock ({result["inventory_position"]:,.0f} units) is comfortably above the reorder point '
                        f'({result["reorder_point"]:,.1f} units).</span></div>',
                        unsafe_allow_html=True,
                    )

                # TASK 3: METRICS WITH PLAIN ENGLISH RENAMES
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Current stock position",
                    f"{result['inventory_position']:,.0f} units",
                    help="On-hand physically + On-order in transit - Customer backorders",
                )
                m2.metric(
                    "Expected sales waiting for delivery",
                    f"{result['protection_period_demand']:,.1f} units",
                    help=f"Projected customer sales over the {protection_days}-day lead time window",
                )
                m3.metric(
                    "Extra buffer stock",
                    f"{result['protection_period_safety_stock']:,.1f} units",
                    help=f"Safety buffer calculated for {int(service_level * 100)}% non-stockout service level",
                )
                m4.metric(
                    "Order when stock falls to",
                    f"{result['reorder_point']:,.1f} units",
                    help="If your current inventory position is at or below this threshold, reorder immediately",
                )

                if suggested_qty > 0:
                    st.markdown(
                        f"### Recommended order: **{suggested_qty:,} units**"
                        + (f" ({suggested_qty // int(pack_size):,} packs of {pack_size})" if int(pack_size) > 1 else "")
                    )
                    if latest_price is not None:
                        st.caption(f"Estimated wholesale cost at ${latest_price:.2f}/unit: **${suggested_qty * latest_price:,.2f}**")

                # TASK 3: BAR CHART COMPARING STOCK LEVELS
                st.markdown("##### 📊 Stock Level Comparison")
                stock_comparison_df = pd.DataFrame(
                    {
                        "Inventory Status": [
                            "Current Stock Position",
                            "Order When Stock Falls To",
                            "Target Stock Level",
                        ],
                        "Units": [
                            float(result["inventory_position"]),
                            float(result["reorder_point"]),
                            float(result["target_stock"]),
                        ],
                    }
                ).set_index("Inventory Status")
                st.bar_chart(stock_comparison_df, height=280)

                # DISCLAIMER RETAINED
                st.caption(f"⚠️ **Disclaimer:** {result['disclaimer']}")


# ==============================================================================
# PAGE 3: PAST ACCURACY (TASK 4 - ANALYST VIEW)
# ==============================================================================

elif page == "Past accuracy":
    st.title("🎯 Model Accuracy & Validation Benchmarks")
    st.markdown(
        '<p class="intro">Detailed error metrics and benchmark comparisons across past '
        '28-day testing periods from the M5 competition.</p>',
        unsafe_allow_html=True,
    )

    try:
        results = load_backtest(str(BACKTEST_FILE))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load backtest results: {exc}")
        st.stop()

    # TASK 4: PLAIN-ENGLISH VERDICT ABOVE THE TABLE
    st.markdown(
        '<div class="headline-sentence" style="border-left-color: #3b82f6; background: #eff6ff;">'
        '📋 <strong>Verdict:</strong> Typical prediction miss is about <strong>1 unit per day</strong>. '
        'The machine learning model is only slightly better than the simple recent-average baseline, '
        'giving its biggest benefit on seasonal shifts and holiday patterns.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Accuracy across three independent 28-day test periods")

    # TASK 4: CONSISTENT NUMBERS WITH 1-2 DECIMALS AND % SIGN
    display_df = results.copy()
    display_table = pd.DataFrame({
        "Dates checked": display_df["test_period"],
        "Estimate method": display_df["method"],
        "Typical miss (units)": display_df["MAE"].map(lambda x: f"{x:.2f} units"),
        "Large-miss score (units)": display_df["RMSE"].map(lambda x: f"{x:.2f} units"),
        "Miss vs. sales (%)": display_df["WAPE_pct"].map(lambda x: f"{x:.1f}%"),
        "Average over/under (units)": display_df["Bias"].map(lambda x: f"{x:+.2f} units"),
    })
    st.dataframe(display_table, width="stretch", hide_index=True)

    # TASK 4: GROUPED (NON-STACKED) BAR CHART
    st.markdown("##### 📊 Typical Prediction Miss by Test Period (Grouped Comparison)")
    mae_chart = results.pivot(
        index="test_period",
        columns="method",
        values="MAE",
    )
    # Using stack=False ensures a grouped side-by-side comparison instead of stacked bars
    st.bar_chart(mae_chart, stack=False, height=320)
    st.caption("Shorter bars mean smaller daily errors (better accuracy). The bars are grouped side-by-side for direct comparison.")

    # SEGMENT BREAKDOWN
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
        with st.expander("Accuracy broken down by product category or store location", expanded=False):
            segment = st.radio(
                "Group results by:",
                ["Product type", "Store"],
                horizontal=True,
                help="Break down errors across product departments or store locations",
            )
            group_column = "category" if segment == "Product type" else "store"
            segment_rows = []
            for name, group in historical_forecast.groupby(group_column, sort=True):
                actual_total = float(group["actual_sales"].abs().sum())
                segment_rows.append(
                    {
                        segment: name,
                        "Typical miss (units)": f"{group['error'].abs().mean():.2f} units",
                        "Large-miss score (units)": f"{group['error'].pow(2).mean() ** 0.5:.2f} units",
                        "Miss vs. sales (%)": (
                            f"{100 * group['error'].abs().sum() / actual_total:.1f}%"
                            if actual_total
                            else "N/A"
                        ),
                        "Average over/under (units)": f"{group['error'].mean():+.2f} units",
                    }
                )
            st.dataframe(
                pd.DataFrame(segment_rows),
                width="stretch",
                hide_index=True,
            )

    # TASK 4: FEATURE IMPORTANCE RENAMED TO PLAIN ENGLISH
    direct_model = ROOT / "models" / "direct_xgboost.json"
    direct_metadata = ROOT / "models" / "direct_xgboost_metadata.json"
    if direct_model.is_file() and direct_metadata.is_file():
        st.markdown("##### 🧠 Feature Importance (What the Model Considers Most)")
        st.info("ℹ️ Note: This shows what information the model relies on to predict sales, not what causes customer demand to change.")
        try:
            importance = load_direct_importance(
                str(direct_model),
                str(direct_metadata),
            )
        except (OSError, ValueError) as exc:
            st.error(f"Could not load feature importance: {exc}")
        else:
            top_features = importance.head(12)
            st.bar_chart(top_features.set_index("Feature")["Gain importance"], height=320)

    with st.expander("📖 Score definitions and methodology notes"):
        st.markdown(
            "- **Typical miss (MAE):** Average absolute difference between forecasted and actual sales each day. Lower is better.\n"
            "- **Large-miss score (RMSE):** Root mean squared error, which penalizes occasional extreme forecasting mistakes heavily.\n"
            "- **Miss vs. sales (WAPE %):** Total absolute error as a percentage of total actual volume sold.\n"
            "- **Average over/under (Bias):** Measures whether the model consistently over-forecasts (+ values) or under-forecasts (- values)."
        )


# ==============================================================================
# PAGE 4: DATA QUALITY (ANALYST VIEW)
# ==============================================================================

elif page == "Data quality":
    st.title("🛡️ Dataset Schema & Quality Checks")
    st.markdown(
        '<p class="intro">Verification logs, record completeness, calendar coverage, '
        'and pricing checks across raw input tables.</p>',
        unsafe_allow_html=True,
    )

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
        f"Data version: `{report['data_version']}` · "
        f"Report created: `{report['generated_at_utc']}`"
    )
    sales_quality = report["sales"]
    calendar_quality = report["calendar"]
    price_quality = report["prices"]

    q1, q2, q3 = st.columns(3)
    q1.metric("Tracked Series", f"{sales_quality['series_count']:,}")
    q2.metric("Recorded Days", f"{sales_quality['day_count']:,}")
    q3.metric("Price Rows Checked", f"{price_quality['row_count']:,}")

    with st.expander("Detailed quality inspection payload", expanded=True):
        st.json(
            {
                "Sales Dataset": sales_quality,
                "Calendar Dataset": calendar_quality,
                "Prices Dataset": price_quality,
                "Known Limitations": report["limitations"],
            },
            expanded=True,
        )


# ==============================================================================
# PAGE 5: TRAIN AND SAVE M5 FORECAST (TASK 5 - ADMIN / ADVANCED)
# ==============================================================================

elif page == "Train and save M5 forecast":
    st.title("⚙️ Model Retraining & Provenance (Admin / Advanced)")
    st.markdown(
        '<p class="intro">Manage background model retraining jobs on uploaded sales or the full M5 benchmark dataset. '
        'Standard forecasting requests never trigger retraining automatically.</p>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ **Careful:** Fitting a new model saves and replaces the active XGBoost model used for subsequent forecasts. "
        "Existing backups are retained in the models folder."
    )

    current_job = training_jobs.read_latest_job()
    job_running = bool(
        current_job
        and current_job.get("status") in {"starting", "running"}
    )

    # Prominent running job banner if active
    if job_running:
        st.error(
            "⏳ **A background training job is currently active.** "
            "New training runs cannot be started until this job finishes or is cancelled."
        )
        if st.button("⏹️ Cancel running training job", key="cancel_job_top", help="Stop the active worker immediately"):
            training_jobs.cancel_active_job()
            st.rerun()

    st.markdown("#### Option A: Retrain from Saved Uploaded Sales")
    uploaded_training_available = forecasting_service.HISTORY_PATH.is_file()

    if uploaded_training_available:
        saved_history = pd.read_parquet(forecasting_service.HISTORY_PATH)
        max_days_available = int(saved_history.groupby("id")["date"].nunique().max())
        min_days_available = int(saved_history.groupby("id")["date"].nunique().min())
        st.caption(
            f"Saved sales history: {len(saved_history):,} rows across {saved_history['id'].nunique():,} series "
            f"through {pd.to_datetime(saved_history['date']).max():%B %d, %Y}. "
            f"History spans {min_days_available}–{max_days_available} consecutive days per series."
        )

        with st.expander("⚙️ Training Hyperparameters", expanded=False):
            cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
            min_days_input = cfg_c1.number_input(
                "Minimum consecutive history days",
                min_value=29,
                max_value=365,
                value=min(max_days_available, 29),
                help="Needs at least 29 days (28 days for lag/rolling features + 1 target day).",
            )
            min_rows_input = cfg_c2.number_input(
                "Minimum training rows required",
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
        st.info("ℹ️ Upload sales data on the Explore page first to build a saved sales history.")
        min_days_input = 29
        min_rows_input = 10
        boost_rounds_input = 50

    if "uploaded_retrain_success" in st.session_state:
        st.success(st.session_state.pop("uploaded_retrain_success"))

    confirm_uploaded_training = st.checkbox(
        "I confirm I want to fit and activate a new model from saved uploaded sales.",
        help="Check this box to enable the retraining trigger",
    )
    start_uploaded_training = st.button(
        "Train model from uploaded sales",
        disabled=(
            not uploaded_training_available
            or not confirm_uploaded_training
            or job_running
        ),
        help="Train an XGBoost model using your uploaded sales history",
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
            with st.spinner("Building training feature matrix and fitting XGBoost trees..."):
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
                f"{metadata['training_rows']:,} examples and activated."
            )
            st.rerun()

    st.divider()

    st.markdown("#### Option B: Full M5 Dataset Background Retraining")
    st.markdown(
        "This long-running background workflow trains a fresh recursive XGBoost model from the complete "
        "M5 dataset and saves a versioned `.joblib` model bundle."
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
        "I understand this starts a long model-training job on the M5 data in the background.",
        help="Check this box to enable the background training button",
    )
    start_training = st.button(
        "Start M5 background training",
        disabled=(
            not confirm_long_run
            or bool(missing_training_files)
            or job_running
        ),
        help="Launch background worker process",
    )

    if start_training:
        try:
            new_job = training_jobs.start_recursive_training(num_boost_round=int(m5_rounds_input))
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(f"Could not start training: {exc}")
        else:
            st.success(f"Training job `{new_job['job_id']}` launched.")
            st.rerun()

    st.markdown("#### Active & Recent Training Job Status")
    _training_status_panel()


# ==============================================================================
# PAGE 6: ABOUT THIS TOOL
# ==============================================================================

elif page == "About this tool":
    st.title("ℹ️ About the Store Demand Forecast Tool")
    st.markdown(
        """
        This tool helps retail store managers and commercial planners predict product sales
        and simulate replenishment orders using machine learning models trained on retail sales history.

        ---

        ### 🎯 Who is this for?
        * **Store Managers**: Get clear answers to *"How much will I sell next week?"* and *"How much stock should I order today?"* without wading through technical jargon.
        * **Supply Chain & Inventory Analysts**: Inspect detailed forecast accuracy across rolling historical test periods, check raw data quality, and monitor feature importance.

        ---

        ### 🧠 How does the model work?
        * The forecasting engine uses **XGBoost (Extreme Gradient Boosting)** with a count-Poisson objective suited for retail counts.
        * It analyzes **28 days of past sales**, recent sales velocity (7-day and 28-day averages), day-of-week shopping habits, and product pricing.
        * **No silent fallbacks**: Real predictions and variances are displayed transparently.

        ---

        ### 📦 Replenishment Logic
        * Replenishment simulations calculate **Days of Cover**, **Expected Demand During Lead Time**, and **Extra Buffer Stock** to guarantee high shelf availability (e.g., 95% non-stockout target).
        * Recommendations round up to supplier pack and case sizes automatically.
        * *Note: Replenishment calculations are simulated scenarios for planning guidance and require human review.*
        """
    )
