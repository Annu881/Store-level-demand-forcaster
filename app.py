"""Simple Streamlit dashboard for exploring saved demand forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_READY = ROOT / "data" / "model_ready"
FORECAST_FILES = {
    "Final 28-day projection in the example data": (
        MODEL_READY / "future_forecast_d1942_d1969.parquet",
        False,
    ),
    "Past 28-day accuracy check": (
        MODEL_READY / "final_forecast_d1914_d1941.parquet",
        True,
    ),
}
BACKTEST_FILE = MODEL_READY / "rolling_backtest_metrics.csv"
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
        ["Explore sales estimates", "Past accuracy", "About this tool"],
        label_visibility="collapsed",
    )

if page == "Explore sales estimates":
    selected_forecast = st.selectbox("What would you like to explore?", list(FORECAST_FILES))
    forecast_path, has_actuals = FORECAST_FILES[selected_forecast]
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

        **How to use it**

        Choose a date range, then select a product type and store. Choose one product
        to see its daily sales estimate, or select all products to see the combined
        demand for your filters.
        """
    )
