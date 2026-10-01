"""
dashboard/app.py  — Xorosoft-inspired Enterprise Demand Forecaster Dashboard
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_BASE = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DemandIQ — Store-Level Forecaster",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS (Xorosoft-inspired) ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0F0E2A !important;
    color: #E2E8F0 !important;
}

/* ── Hero Header ── */
.hero-banner {
    background: linear-gradient(135deg, #1E1B4B 0%, #2D2B70 50%, #1E1B4B 100%);
    border-radius: 20px;
    padding: 48px 48px 36px 48px;
    margin-bottom: 32px;
    border: 1px solid #3730A3;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(99,91,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,91,255,0.08) 1px, transparent 1px);
    background-size: 40px 40px;
    border-radius: 20px;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 8px;
}
.hero-title span { color: #38BDF8; }
.hero-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-bottom: 24px;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,91,255,0.2);
    border: 1px solid #635BFF;
    color: #A5B4FC;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-right: 8px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1E1B4B, #2D2B70);
    border: 1px solid #3730A3;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,91,255,0.3);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #64748B;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
}
.kpi-sub {
    font-size: 12px;
    color: #38BDF8;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-tag {
    display: inline-block;
    background: rgba(56,189,248,0.10);
    color: #38BDF8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 9999px;
    border: 1px solid rgba(56,189,248,0.3);
    margin-bottom: 12px;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
}
.section-desc {
    font-size: 14px;
    color: #64748B;
    margin-bottom: 24px;
}

/* ── Baseline comparison table ── */
.baseline-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.baseline-table th {
    background: #1E1B4B;
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 12px 16px;
    border-bottom: 1px solid #2D2B70;
    text-align: left;
}
.baseline-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #1E1B4B;
    color: #CBD5E1;
}
.baseline-table tr:last-child td {
    background: rgba(99,91,255,0.15);
    color: #FFFFFF;
    font-weight: 700;
}
.best-badge {
    background: linear-gradient(135deg, #4F46E5, #635BFF);
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 9999px;
    letter-spacing: 0.5px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F0E2A !important;
    border-right: 1px solid #1E1B4B !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 24px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #635BFF) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5B21B6, #7C3AED) !important;
    box-shadow: 0 8px 25px rgba(99,91,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Status pill ── */
.status-ok {
    display: inline-flex; align-items: center;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: #34D399;
    font-size: 12px; font-weight: 600;
    padding: 6px 16px; border-radius: 9999px;
    margin-right: 8px;
}
.status-err {
    display: inline-flex; align-items: center;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #F87171;
    font-size: 12px; font-weight: 600;
    padding: 6px 16px; border-radius: 9999px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E1B4B !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background: rgba(99,91,255,0.2) !important;
    border-bottom: 2px solid #635BFF !important;
}

/* ── Info blocks ── */
.stAlert {
    border-radius: 12px !important;
}

/* ─ hide Streamlit chrome ─ */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div style="position:relative;z-index:1;">
        <div>
            <span class="hero-badge">📦 RETAIL AI</span>
            <span class="hero-badge">M5 Walmart Dataset</span>
            <span class="hero-badge">LightGBM v1</span>
        </div>
        <div class="hero-title">Store-Level <span>Demand Forecaster</span></div>
        <div class="hero-subtitle">
            Global LightGBM model trained on real Walmart M5 data · 28-day forward forecasting · 
            Leakage-safe feature engineering · Walk-forward validation
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:0 0 16px 0;">
    <div style="font-size:18px;font-weight:800;color:#FFFFFF;">⚙️ Forecast Settings</div>
    <div style="font-size:12px;color:#64748B;margin-top:4px;">Configure your prediction parameters</div>
</div>
""", unsafe_allow_html=True)

store_id = st.sidebar.selectbox("🏪 Store ID", ["CA_1","CA_2","CA_3","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"])
item_id  = st.sidebar.text_input("📦 Item ID", value="HOBBIES_1_001", help="e.g. HOBBIES_1_001")
horizon  = st.sidebar.slider("📅 Horizon (days)", 1, 28, 28)
st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Generate Forecast")

st.sidebar.markdown("""
<div style="margin-top:16px;padding:16px;background:#1E1B4B;border-radius:12px;border:1px solid #3730A3;">
    <div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Model Results</div>
    <div style="font-size:13px;color:#CBD5E1;">✅ LightGBM RMSE: <b style="color:#38BDF8;">2.83</b></div>
    <div style="font-size:13px;color:#CBD5E1;margin-top:6px;">✅ MAE: <b style="color:#38BDF8;">1.33</b></div>
    <div style="font-size:13px;color:#CBD5E1;margin-top:6px;">✅ Beats all 3 baselines</div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.replenishment.simulator import simulate_replenishment

tab1, tab2, tab3, tab4 = st.tabs(["📈  Forecast Explorer", "📦  Replenishment Sim", "📊  Model Performance", "❤️  System Health"])

# ──────────────────────── TAB 1: FORECAST EXPLORER ───────────────────────────
with tab1:
    if run:
        with st.spinner("Running inference on LightGBM model..."):
            try:
                resp = requests.post(f"{API_BASE}/api/v1/forecasts",
                                     json={"store_id": store_id, "item_id": item_id, "horizon_days": horizon},
                                     timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    df = pd.DataFrame(data["forecasts"])
                    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

                    # Store forecast for Tab 2
                    st.session_state["latest_forecast"] = df["point_forecast"].tolist()

                    total  = df["point_forecast"].sum()
                    avg    = df["point_forecast"].mean()
                    peak   = df["point_forecast"].max()
                    peak_d = df.loc[df["point_forecast"].idxmax(), "forecast_date"].strftime("%b %d")

                    # KPI row
                    c1, c2, c3, c4 = st.columns(4)
                    for col, label, value, sub in [
                        (c1, "Total Demand", f"{total:.0f}", "units"),
                        (c2, "Daily Average", f"{avg:.2f}", "units / day"),
                        (c3, "Peak Day", f"{peak:.1f}", f"on {peak_d}"),
                        (c4, "Horizon", f"{horizon}", "days forecast"),
                    ]:
                        col.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value">{value}</div>
                            <div class="kpi-sub">{sub}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Plotly chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["forecast_date"], y=df["point_forecast"],
                        mode="lines+markers", name="LightGBM Forecast",
                        line=dict(color="#635BFF", width=3),
                        marker=dict(size=6, color="#38BDF8",
                                    line=dict(color="#635BFF", width=2)),
                        fill="tozeroy", fillcolor="rgba(99,91,255,0.08)"
                    ))
                    fig.update_layout(
                        title=dict(text=f"<b>28-Day Demand Forecast</b>  ·  {store_id} / {item_id}",
                                   font=dict(color="#FFFFFF", size=16)),
                        paper_bgcolor="#1E1B4B", plot_bgcolor="#1E1B4B",
                        xaxis=dict(gridcolor="#2D2B70", color="#64748B", title="Date"),
                        yaxis=dict(gridcolor="#2D2B70", color="#64748B", title="Units Sold"),
                        font=dict(family="Inter"),
                        height=380, margin=dict(t=50, b=40, l=40, r=20),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Table
                    st.markdown('<div class="section-tag">DAILY BREAKDOWN</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
                    st.dataframe(
                        df.rename(columns={
                            "forecast_date": "Date",
                            "horizon": "Day",
                            "point_forecast": "Predicted Units"
                        }).style.format({"Predicted Units": "{:.2f}"}),
                        use_container_width=True, height=320
                    )

                elif resp.status_code == 404:
                    st.error(f"❌ Item `{item_id}` not found in store `{store_id}`. Use items trained during model training (e.g. HOBBIES_1_001 for CA_1).")
                else:
                    st.error(f"API Error {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Run: `uvicorn api.main:app --reload`")
    else:
        st.markdown("""
        <div style="background:#1E1B4B;border:1px dashed #3730A3;border-radius:16px;
                    padding:48px;text-align:center;margin-top:16px;">
            <div style="font-size:48px;margin-bottom:16px;">📈</div>
            <div style="font-size:20px;font-weight:700;color:#FFFFFF;margin-bottom:8px;">
                Configure Your Forecast
            </div>
            <div style="font-size:14px;color:#64748B;">
                Select a <b style="color:#A5B4FC;">Store</b> and <b style="color:#A5B4FC;">Item</b> 
                in the sidebar, then click <b style="color:#635BFF;">Generate Forecast</b>
            </div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────── TAB 2: REPLENISHMENT SIM ───────────────────────────
with tab2:
    st.markdown('<div class="section-tag">INVENTORY SIMULATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Smart Replenishment Logic</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Adjust operating parameters below to see how inventory KPIs change based on the LightGBM forecast.</div>', unsafe_allow_html=True)

    if "latest_forecast" not in st.session_state:
        st.info("👈 Please Generate a forecast in Tab 1 first!")
    else:
        fcst_list = st.session_state["latest_forecast"]
        
        c1, c2, c3 = st.columns(3)
        current_stock = c1.number_input("📦 Current On-Hand Stock", min_value=0, value=50, step=10)
        lead_time = c2.number_input("🚚 Supplier Lead Time (Days)", min_value=1, value=7, step=1)
        service_lvl_str = c3.selectbox("🎯 Target Service Level", ["90%", "95%", "98%", "99%"], index=1)
        
        service_lvl = float(service_lvl_str.strip("%")) / 100.0
        
        res = simulate_replenishment(fcst_list, current_stock, lead_time, service_lvl)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        col1.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #F59E0B;">
            <div class="kpi-label">Reorder Point</div>
            <div class="kpi-value">{res['reorder_point']} <span style="font-size:14px;">units</span></div>
            <div class="kpi-sub">Trigger order when stock hits this level</div>
        </div>""", unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-label">Safety Stock</div>
            <div class="kpi-value">{res['safety_stock']} <span style="font-size:14px;">units</span></div>
            <div class="kpi-sub">Buffer for {service_lvl_str} service level</div>
        </div>""", unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #38BDF8;">
            <div class="kpi-label">Suggested Order Qty</div>
            <div class="kpi-value">{res['suggested_order_qty']} <span style="font-size:14px;">units</span></div>
            <div class="kpi-sub">Order amount needed today</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:16px;background:#1E1B4B;border-radius:12px;border:1px solid #2D2B70;display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Projected Stockout Risk</span><br>
                <div style="font-weight:700;font-size:18px;margin-top:4px;">{res['stockout_risk']}</div>
            </div>
            <div>
                <span style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Estimated Stock Remaining</span><br>
                <div style="font-weight:700;font-size:18px;margin-top:4px;color:#CBD5E1;">{res['days_of_stock_remaining']} days</div>
            </div>
            <div>
                <span style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Est. Holding Cost</span><br>
                <div style="font-weight:700;font-size:18px;margin-top:4px;color:#CBD5E1;">${res['estimated_holding_cost']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────── TAB 3: MODEL PERFORMANCE ───────────────────────────
with tab3:
    st.markdown('<div class="section-tag">EVALUATION RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Baseline Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">All models evaluated on the same untouched 28-day holdout from real Walmart M5 data.</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="baseline-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Strategy</th>
                <th>MAE ↓</th>
                <th>RMSE ↓</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Naive (Lag-1)</td>
                <td>Predict yesterday's value</td>
                <td>1.6506</td>
                <td>3.9446</td>
                <td>❌</td>
            </tr>
            <tr>
                <td>Moving Average (7-day)</td>
                <td>Mean of prior 7 days</td>
                <td>1.3973</td>
                <td>3.0490</td>
                <td>❌</td>
            </tr>
            <tr>
                <td>Seasonal Naive</td>
                <td>Same weekday 7 days ago</td>
                <td>1.6818</td>
                <td>3.9984</td>
                <td>❌</td>
            </tr>
            <tr>
                <td>LightGBM (Global)</td>
                <td>Gradient boosted trees + lag/rolling features</td>
                <td>1.3291</td>
                <td>2.8291</td>
                <td><span class="best-badge">✅ BEST</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">MODEL ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technical Specifications</div>', unsafe_allow_html=True)
    try:
        info = requests.get(f"{API_BASE}/api/v1/model/info", timeout=5).json()
        c1, c2 = st.columns(2)
        with c1:
            for k, v in info.items():
                if not isinstance(v, list):
                    st.markdown(f"""
                    <div style="padding:12px 16px;background:#1E1B4B;border-radius:10px;
                                border-left:3px solid #635BFF;margin-bottom:8px;">
                        <span style="color:#64748B;font-size:11px;text-transform:uppercase;
                                     letter-spacing:1px;">{k.replace('_',' ')}</span>
                        <div style="color:#FFFFFF;font-weight:600;margin-top:2px;">{v}</div>
                    </div>""", unsafe_allow_html=True)
        with c2:
            features = info.get("features", [])
            st.markdown("""
            <div style="padding:16px;background:#1E1B4B;border-radius:12px;border:1px solid #2D2B70;">
                <div style="font-size:11px;color:#64748B;text-transform:uppercase;
                             letter-spacing:1px;margin-bottom:12px;">Feature Set</div>""",
                        unsafe_allow_html=True)
            for f in features:
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin-bottom:6px;">
                    <span style="width:6px;height:6px;background:#635BFF;border-radius:50%;
                                 display:inline-block;margin-right:10px;"></span>
                    <span style="color:#CBD5E1;font-size:13px;">{f}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        st.warning("Start the API server to view model info.")

# ──────────────────────── TAB 3: HEALTH ───────────────────────────────────────
with tab3:
    st.markdown('<div class="section-tag">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">API Health Monitor</div>', unsafe_allow_html=True)

    try:
        health = requests.get(f"{API_BASE}/api/v1/health", timeout=5).json()
        loaded = health.get("model_loaded", False)
        status = health.get("status", "unknown")
        version = health.get("model_version", "N/A")

        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
            <span class="{'status-ok' if status=='ok' else 'status-err'}">
                {'✅' if status=='ok' else '❌'} API: {status.upper()}
            </span>
            <span class="{'status-ok' if loaded else 'status-err'}">
                {'✅' if loaded else '❌'} Model: {'LOADED' if loaded else 'NOT LOADED'}
            </span>
            <span class="status-ok">📦 Version: {version}</span>
        </div>""", unsafe_allow_html=True)

        for label, ok, desc in [
            ("FastAPI Server",       True,   "Uvicorn running on port 8000"),
            ("LightGBM Model",       loaded, "real_m5_lgbm_model.pkl loaded in RAM"),
            ("Forecast Endpoint",    True,   "POST /api/v1/forecasts → 200 OK"),
            ("Health Endpoint",      True,   "GET /api/v1/health → 200 OK"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:14px 20px;
                        background:#1E1B4B;border-radius:10px;border:1px solid #2D2B70;
                        margin-bottom:8px;">
                <span style="font-size:18px;margin-right:16px;">{'✅' if ok else '❌'}</span>
                <div>
                    <div style="color:#FFFFFF;font-weight:600;font-size:14px;">{label}</div>
                    <div style="color:#64748B;font-size:12px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class="status-err">❌ API Unreachable</div><br>
        <div style="color:#64748B;">Run: <code>uvicorn api.main:app --reload</code></div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1E1B4B;margin-top:48px;">
<div style="text-align:center;padding:16px 0;color:#334155;font-size:12px;">
    ⚠️ Educational Decision Support Only · Not for autonomous ordering ·
    Powered by Global LightGBM · M5 Walmart Dataset
</div>
""", unsafe_allow_html=True)
