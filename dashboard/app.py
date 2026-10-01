"""
dashboard/app.py
Streamlit dashboard for the Store-Level Demand Forecaster.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Store-Level Demand Forecaster",
    page_icon="📦",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📦 Store-Level Demand Forecaster")
st.caption("Powered by Global LightGBM trained on Walmart M5 data | Educational Decision Support Only")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Forecast Settings")

store_id = st.sidebar.selectbox(
    "Store ID",
    ["CA_1", "CA_2", "CA_3", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"],
    index=0
)

item_id = st.sidebar.text_input(
    "Item ID",
    value="HOBBIES_1_001",
    help="Enter a valid M5 item ID, e.g. HOBBIES_1_001"
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1, max_value=28, value=28
)

run_forecast = st.sidebar.button("🚀 Generate Forecast", type="primary")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast Explorer", "📊 Model Info", "❤️ Health"])

# ── TAB 1: Forecast Explorer ──────────────────────────────────────────────────
with tab1:
    if run_forecast:
        with st.spinner(f"Running {horizon}-day forecast for {store_id} / {item_id}..."):
            try:
                resp = requests.post(f"{API_BASE}/api/v1/forecasts", json={
                    "store_id": store_id,
                    "item_id": item_id,
                    "horizon_days": horizon
                }, timeout=60)

                if resp.status_code == 200:
                    data = resp.json()
                    forecasts = data["forecasts"]
                    df = pd.DataFrame(forecasts)
                    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

                    # KPI cards
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predicted Demand", f"{df['point_forecast'].sum():.0f} units")
                    col2.metric("Avg Daily Forecast", f"{df['point_forecast'].mean():.2f} units")
                    col3.metric("Forecast Horizon", f"{horizon} days")

                    # Forecast chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["forecast_date"],
                        y=df["point_forecast"],
                        mode="lines+markers",
                        name="LightGBM Forecast",
                        line=dict(color="#00b4d8", width=2),
                        marker=dict(size=5)
                    ))
                    fig.update_layout(
                        title=f"28-Day Demand Forecast: {store_id} / {item_id}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Units Sold",
                        template="plotly_dark",
                        height=420
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    st.subheader("📋 Daily Forecast Table")
                    st.dataframe(
                        df.rename(columns={
                            "forecast_date": "Date",
                            "horizon": "Horizon (Day)",
                            "point_forecast": "Predicted Units"
                        }),
                        use_container_width=True
                    )

                elif resp.status_code == 404:
                    st.error(f"❌ Item not found: {item_id} in store {store_id}. Try CA_1 stores with items like HOBBIES_1_001.")
                else:
                    st.error(f"❌ API Error {resp.status_code}: {resp.json().get('detail','Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the FastAPI server is running: `uvicorn api.main:app --reload`")
    else:
        st.info("👈 Select a store, item, and horizon in the sidebar, then click **Generate Forecast**.")

# ── TAB 2: Model Info ─────────────────────────────────────────────────────────
with tab2:
    try:
        info = requests.get(f"{API_BASE}/api/v1/model/info", timeout=5).json()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Model Details")
            st.json(info)
        with col2:
            st.subheader("📐 Evaluation Results (Real M5 Data)")
            st.table(pd.DataFrame([
                {"Model": "Naive (lag-1)",       "MAE": 1.6506, "RMSE": 3.9446},
                {"Model": "Moving Avg (7-day)",  "MAE": 1.3973, "RMSE": 3.0490},
                {"Model": "Seasonal Naive",      "MAE": 1.6818, "RMSE": 3.9984},
                {"Model": "✅ LightGBM (Best)",  "MAE": 1.3291, "RMSE": 2.8291},
            ]))
    except Exception:
        st.error("❌ Cannot load model info. Ensure the FastAPI server is running.")

# ── TAB 3: Health ─────────────────────────────────────────────────────────────
with tab3:
    try:
        health = requests.get(f"{API_BASE}/api/v1/health", timeout=5).json()
        if health["model_loaded"]:
            st.success(f"✅ API Status: **{health['status'].upper()}**")
            st.success(f"✅ Model Loaded: **{health['model_version']}**")
        else:
            st.error("❌ Model not loaded. Train the model first.")
    except Exception:
        st.error("❌ API is not reachable. Start the server with `uvicorn api.main:app --reload`")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ **Disclaimer:** All forecasts are for educational decision support only. Not for autonomous ordering or operational deployment.")
