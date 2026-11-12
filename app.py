"""Simple Streamlit dashboard for exploring saved demand forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from inventory import ReplenishmentInputs, simulate_replenishment


ROOT = Path(__file__).resolve().parent
MODEL_READY = ROOT / "data" / "model_ready"
QUALITY_REPORT = ROOT / "data" / "quality" / "latest.json"
FORECAST_FILES = {
    "Final 28-day projection in the example data": (
        MODEL_READY / "future_forecast_d1942_d1969.parquet",
        False,
    ),
    "Past 28-day accuracy check": (
        MODEL_READY / "final_forecast_d1914_d1941.parquet",
        True,
    ),
    "Direct XGBoost candidate": (
        MODEL_READY / "direct_forecast_d1942_d1969.parquet",
        False,
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
    import json

    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data-quality report not found: {path}. Run data_quality.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def readable_series(series_id: str) -> str:
    parts = series_id.split("_")
    if len(parts) >= 6:
        category = parts[0].title()
        state = {
            "CA": "California",
            "TX": "Texas",
            "WI": "Wisconsin",
        }.get(parts[3], parts[3])
        return (
            f"{category} · Department {parts[1]} · Item {parts[2]} · "
            f"{state}, Store {parts[4]}"
        )
    return series_id


def readable_store(store: str) -> str:
    state, store_number = store.split("_", maxsplit=1)
    state_name = {
        "CA": "California",
        "TX": "Texas",
        "WI": "Wisconsin",
    }.get(state, state)
    return f"{state_name}, Store {store_number}"


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
            "About this tool",
        ],
        label_visibility="collapsed",
    )

if page == "Explore sales estimates":
    available_forecasts = {
        label: details
        for label, details in FORECAST_FILES.items()
        if details[0].is_file()
    }
    if not available_forecasts:
        st.error("No saved forecast files are available.")
        st.stop()
    selected_forecast = st.selectbox(
        "What would you like to explore?",
        list(available_forecasts),
    )
    forecast_path, has_actuals = available_forecasts[selected_forecast]
    try:
        forecast = load_forecast(str(forecast_path), has_actuals)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load forecast data: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("Filters")
        categories = ["All categories"] + sorted(forecast["category"].unique())
        category = st.selectbox(
            "Product type",
            categories,
            format_func=lambda value: (
                "All product types" if value == "All categories" else readable_category(value)
            ),
        )
        filtered = forecast
        if category != "All categories":
            filtered = filtered.loc[filtered["category"] == category]

        stores = ["All stores"] + sorted(filtered["store"].unique())
        store = st.selectbox(
            "Location",
            stores,
            format_func=lambda value: (
                "All locations" if value == "All stores" else readable_store(value)
            ),
        )
        if store != "All stores":
            filtered = filtered.loc[filtered["store"] == store]

        available_series = sorted(filtered["id"].unique())
        all_series_label = "All products in this view"
        series_selection = st.selectbox(
            "Product",
            [all_series_label] + available_series,
            format_func=lambda value: (
                all_series_label if value == all_series_label else readable_series(value)
            ),
        )

    if filtered.empty:
        st.warning("There are no forecast rows for these filters.")
        st.stop()

    if series_selection != all_series_label:
        chart_rows = filtered.loc[filtered["id"] == series_selection].copy()
        accuracy_rows = chart_rows
        chart_title = readable_series(series_selection)
        demand_scope = "Selected product/store"
    else:
        accuracy_rows = filtered
        chart_rows = (
            filtered.groupby(["d", "date"], as_index=False)
            .agg(
                predicted_sales=("predicted_sales", "sum"),
                **(
                    {"actual_sales": ("actual_sales", "sum")}
                    if has_actuals
                    else {}
                ),
            )
        )
        chart_title = "Combined estimated sales for these products"
        demand_scope = "Selected products"

    chart_rows = chart_rows.sort_values("date")
    forecast_units = float(chart_rows["predicted_sales"].sum())
    forecast_daily = float(chart_rows["predicted_sales"].mean())
    period_range = format_date_range(
        chart_rows["date"].min(),
        chart_rows["date"].max(),
    )

    if has_actuals:
        st.subheader(f"Past accuracy check: {period_range}")
        st.caption(
            "These dates are from the past. We compare model estimates with the actual "
            "sales recorded during this period."
        )
    else:
        st.subheader(f"Example projection: {period_range}")
        st.caption(
            "This is the final 28-day period in the supplied example dataset."
        )

    if not has_actuals:
        st.info(
            "This is a historical example from the M5 retail dataset—not a forecast "
            "for today or upcoming dates. The dataset does not include actual sales "
            "for this period, so we cannot check how accurate these estimates were."
        )
    else:
        st.info(
            "This is a past period used to check forecast accuracy. It is not a "
            "prediction for future dates."
        )

    first, second, third = st.columns(3)
    first.metric(
        "Estimated sales over these dates",
        format_units(forecast_units),
        help="Total estimated units summed across all selected products and days.",
    )
    second.metric(
        "Estimated sales per day",
        f"{forecast_daily:,.1f} units",
        help="Average daily estimate for the selected product or group.",
    )
    third.metric(
        "Product and store combinations",
        f"{filtered['id'].nunique():,}" if series_selection == all_series_label else "1",
        help="Number of different products at stores included in this view.",
    )

    st.subheader(chart_title)
    st.caption("Estimated sales by date, measured in units.")
    chart_data = chart_rows.set_index("date")
    if has_actuals:
        chart_data = chart_data[["actual_sales", "predicted_sales"]].rename(
            columns={
                "actual_sales": "Actual sales",
                "predicted_sales": "Predicted sales",
            }
        )
        st.line_chart(chart_data)
        st.caption(
            "Compare the estimated-sales line with the actual-sales line. The closer "
            "they are, the more accurate the estimate was."
        )
    else:
        st.line_chart(chart_data[["predicted_sales"]].rename(
            columns={"predicted_sales": "Predicted sales"}
        ))
        st.caption(
            "This chart shows estimates only. Actual sales for these dates were not "
            "provided in the dataset."
        )

    if has_actuals:
        actual = accuracy_rows["actual_sales"].astype(float)
        predicted = accuracy_rows["predicted_sales"].astype(float)
        error = predicted - actual
        actual_total = float(actual.abs().sum())
        mae = float(error.abs().mean())
        rmse = float((error.pow(2).mean()) ** 0.5)
        wape = float(100 * error.abs().sum() / actual_total) if actual_total else float("nan")
        bias = float(error.mean())
        st.subheader("Accuracy for this selection")
        st.caption(
            "These scores summarize how close the estimates were to actual sales over "
            "the selected past period. Lower error scores are better."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Typical daily miss",
            f"{mae:.2f} units",
            help="Average absolute difference between estimated and actual sales.",
        )
        c2.metric(
            "Score for larger misses",
            f"{rmse:.2f} units",
            help="Like the typical miss, but larger errors count more heavily.",
        )
        c3.metric(
            "Miss compared with sales",
            f"{wape:.1f}%",
            help="Total absolute error as a share of total actual sales.",
        )
        c4.metric(
            "Usual direction of error",
            describe_bias(bias),
            help="Shows whether estimated sales were usually below or above actual sales.",
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
            str(FORECAST_FILES["Past 28-day accuracy check"][0]),
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
    available_forecasts = {
        label: details
        for label, details in FORECAST_FILES.items()
        if details[0].is_file() and not details[1]
    }
    if not available_forecasts:
        st.error("No future forecast is available for this simulation.")
        st.stop()
    selected_forecast = st.selectbox(
        "Forecast to use",
        list(available_forecasts),
        key="replenishment_forecast",
    )
    forecast_path, has_actuals = available_forecasts[selected_forecast]
    try:
        forecast = load_forecast(str(forecast_path), has_actuals)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load forecast data: {exc}")
        st.stop()

    series = (
        forecast[["store", "id"]]
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
    with st.form("replenishment_inputs"):
        first, second, third = st.columns(3)
        on_hand = first.number_input("Units currently on hand", min_value=0.0, value=0.0)
        on_order = second.number_input("Units already on order", min_value=0.0, value=0.0)
        backorders = third.number_input("Units backordered", min_value=0.0, value=0.0)
        fourth, fifth, sixth = st.columns(3)
        lead_time = fourth.number_input(
            "Supplier lead time (days)", min_value=1, max_value=28, value=7
        )
        review_period = fifth.number_input(
            "Review period (days)", min_value=0, max_value=27, value=0
        )
        pack_size = sixth.number_input("Units per order pack", min_value=1, value=1)
        seventh, eighth = st.columns(2)
        minimum_order = seventh.number_input("Minimum order quantity", min_value=0, value=0)
        service_level = eighth.selectbox(
            "Target service level",
            [0.8, 0.9, 0.95, 0.99],
            index=2,
        )
        daily_residual_std = st.number_input(
            "Daily forecast-error variability (units)",
            min_value=0.0,
            value=0.0,
            help="Optional estimate; zero means this simulation adds no safety stock.",
        )
        submitted = st.form_submit_button("Run simulation")

    if submitted:
        protection_days = int(lead_time + review_period)
        selected = forecast.loc[forecast["id"] == selected_id].sort_values("date")
        daily_forecasts = selected["predicted_sales"].astype(float).tolist()
        if protection_days > len(daily_forecasts):
            st.error("Lead time plus review period exceeds the available forecast horizon.")
        else:
            try:
                result = simulate_replenishment(
                    daily_forecasts,
                    ReplenishmentInputs(
                        on_hand=on_hand,
                        on_order=on_order,
                        backorders=backorders,
                        lead_time_days=int(lead_time),
                        review_period_days=int(review_period),
                        pack_size=int(pack_size),
                        minimum_order_quantity=int(minimum_order),
                        service_level=service_level,
                        daily_residual_std=daily_residual_std,
                    ),
                )
            except ValueError as exc:
                st.error(f"Could not run simulation: {exc}")
            else:
                first, second, third = st.columns(3)
                first.metric(
                    "Inventory position",
                    f"{result['inventory_position']:,.1f} units",
                )
                second.metric("Reorder point", f"{result['reorder_point']:,.1f} units")
                third.metric(
                    "Simulated order quantity",
                    f"{result['suggested_order_quantity']:,} units",
                )
                st.write(result["severity"])
                st.caption(result["disclaimer"])

elif page == "Data quality":
    st.subheader("Dataset checks")
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

elif page == "About this tool":
    st.subheader("About this sales-estimate tool")
    st.markdown(
        """
        This tool demonstrates how a computer model can estimate daily sales for
        different products at different stores, using the M5 retail dataset.

        **What the dashboard shows**
        - Estimates compared with real recorded sales for an earlier test period.
        - An additional 28-day estimate for the last dates in the supplied example data.
        - A comparison between the model and a simple recent-sales average.

        **Important limitation**

        The extra 28-day estimate covers May 23–June 19, 2016. These are historical
        dataset dates, not today's or upcoming dates. Actual sales for that period
        were not supplied, so we cannot measure the accuracy of this estimate.

        XGBoost was retained because the saved recursive comparison reported a
        lower recursive MAE than LightGBM and the two-stage candidate. A separate
        direct XGBoost candidate follows the requested 1–28-day design and must be
        trained and evaluated fairly before replacing the recursive model.

        **How to use it**

        Choose a date range, then select a product type and store. Choose one product
        to see its daily sales estimate, or select all products to see the combined
        demand for your filters.
        """
    )
