"""
╔══════════════════════════════════════════════════════════════╗
║  Walmart M5 Store Sales Forecasting — Streamlit Dashboard     ║
║  LightGBM · Prophet · Hierarchical Reconciliation · WRMSSE    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.pipeline import ForecastingPipeline

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="M5 Forecasting | Walmart",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --walmart-blue: #0071CE;
    --walmart-yellow: #FFC220;
    --bg-dark: #0D1117;
    --bg-card: #161B22;
    --bg-card2: #1C2128;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --green: #3FB950;
    --red: #F85149;
    --orange: #D29922;
}

.stApp { background: var(--bg-dark); }
.main .block-container { max-width: 1400px; padding: 1.5rem 2rem; }

/* Typography */
h1, h2, h3, h4, .metric-label { font-family: 'IBM Plex Sans', sans-serif !important; }
code, .stCode { font-family: 'IBM Plex Mono', monospace !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0D1117 0%, #0a1628 40%, #0D1117 100%);
    border: 1px solid var(--walmart-blue);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--walmart-blue), var(--walmart-yellow), var(--walmart-blue));
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: var(--text-primary);
    font-family: 'IBM Plex Sans', sans-serif;
    margin: 0; letter-spacing: -0.02em;
}
.hero-subtitle {
    color: var(--text-secondary); font-size: 0.95rem; margin-top: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
}
.hero-badge {
    display: inline-block; background: rgba(0,113,206,0.15);
    border: 1px solid rgba(0,113,206,0.4); color: var(--walmart-blue);
    padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace; margin-right: 0.5rem; margin-top: 0.6rem;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--walmart-blue); }
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--walmart-blue), var(--walmart-yellow));
}
.metric-value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.metric-label-text { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; }
.metric-delta { font-size: 0.78rem; margin-top: 0.3rem; font-family: 'IBM Plex Mono'; }
.delta-pos { color: var(--green); }
.delta-neg { color: var(--red); }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 0.7rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.1rem; font-weight: 600; color: var(--text-primary);
    padding: 0.8rem 0; margin: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}
.section-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--walmart-blue); flex-shrink: 0;
}

/* Pipeline step tracker */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.step-icon { font-size: 1.5rem; }
.step-name { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem; }
.step-active { border-color: var(--walmart-blue); }
.step-done { border-color: var(--green); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--walmart-blue), #005ea6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.7rem 2rem !important; width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card); border-radius: 8px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: var(--walmart-blue) !important;
    border-bottom: 2px solid var(--walmart-blue);
}

/* Info boxes */
.info-box {
    background: rgba(0,113,206,0.08);
    border: 1px solid rgba(0,113,206,0.3);
    border-radius: 8px; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: var(--text-secondary);
    font-family: 'IBM Plex Mono', monospace;
}
.warn-box {
    background: rgba(255,194,32,0.08);
    border: 1px solid rgba(255,194,32,0.3);
    border-radius: 8px; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #d4a017;
}

/* Select boxes */
.stSelectbox label, .stSlider label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#E6EDF3", size=12),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363D", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

COLORS = {
    "actual": "#E6EDF3",
    "lgbm": "#0071CE",
    "prophet": "#FFC220",
    "naive": "#8B949E",
    "ci": "rgba(0,113,206,0.15)",
}

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# ─────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📦 Walmart M5 Store Sales Forecasting</div>
    <div class="hero-subtitle">
        Hierarchical Time-Series · LightGBM + Prophet · Bottom-Up Reconciliation · WRMSSE Evaluation
    </div>
    <div style="margin-top: 0.8rem;">
        <span class="hero-badge">Python</span>
        <span class="hero-badge">LightGBM</span>
        <span class="hero-badge">Prophet</span>
        <span class="hero-badge">Hierarchical TS</span>
        <span class="hero-badge">WRMSSE</span>
        <span class="hero-badge">3,000+ Products · 10 Stores</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    st.markdown("---")

    st.markdown("**Dataset Scale**")
    n_items = st.slider("Items per department", min_value=3, max_value=12, value=6, step=1,
                        help="Scales number of products. Real M5 has 3,049 items.")
    lgbm_estimators = st.slider("LightGBM estimators", min_value=50, max_value=500, value=150, step=50)
    forecast_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=28, value=28, step=7)

    st.markdown("---")
    st.markdown("**Reconciliation Method**")
    recon_method = st.selectbox("Method", ["bottom_up", "top_down", "ols"],
                                 help="bottom_up: sum item→store→total. ols: regression-based.")

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)

    if st.session_state.trained:
        st.success("✅ Pipeline trained")

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #8B949E; font-family: IBM Plex Mono;'>
    <b>Architecture</b><br>
    • M5 synthetic data generator<br>
    • Wide→long preprocessing<br>
    • 25+ engineered features<br>
    • Global Tweedie LightGBM<br>
    • Per-store Prophet models<br>
    • Bottom-up reconciliation<br>
    • WRMSSE evaluation<br>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────────────────────────────

if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(pct, msg):
        progress_bar.progress(pct)
        status_text.markdown(f"<div class='info-box'>⟳ {msg}</div>", unsafe_allow_html=True)

    pipeline = ForecastingPipeline(
        n_items_per_dept=n_items,
        forecast_horizon=forecast_horizon,
        val_days=forecast_horizon,
        lgbm_n_estimators=lgbm_estimators,
        seed=42,
    )
    pipeline.reconciler.method = recon_method
    pipeline.run(progress_callback=progress_callback)

    st.session_state.pipeline = pipeline
    st.session_state.trained = True
    progress_bar.empty()
    status_text.empty()
    st.rerun()

# ─────────────────────────────────────────────────────────────
# Main Dashboard (shown after training)
# ─────────────────────────────────────────────────────────────

if not st.session_state.trained or st.session_state.pipeline is None:
    # Landing screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="step-icon">📊</div>
            <div class="metric-value" style="font-size:1.4rem; color:#0071CE;">3,049</div>
            <div class="metric-label-text">Products (M5 Scale)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="step-icon">🏪</div>
            <div class="metric-value" style="font-size:1.4rem; color:#FFC220;">10</div>
            <div class="metric-label-text">Walmart Stores (3 states)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="step-icon">📅</div>
            <div class="metric-value" style="font-size:1.4rem; color:#3FB950;">1,941</div>
            <div class="metric-label-text">Days of history</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline architecture diagram
    st.markdown('<div class="section-header"><div class="section-dot"></div>Pipeline Architecture</div>', unsafe_allow_html=True)
    steps = [
        ("🗄️", "Data\nGeneration"),
        ("⚙️", "Preprocessing\n& SQL-like joins"),
        ("🔧", "Feature\nEngineering"),
        ("⚡", "LightGBM\nGlobal Model"),
        ("📈", "Prophet\nPer-Store"),
        ("🔗", "Hierarchical\nReconciliation"),
        ("📐", "WRMSSE\nEvaluation"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, label) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-name">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    ⚡ Configure pipeline parameters in the sidebar and click <b>Run Full Pipeline</b> to start.
    Adjust items/estimators for speed vs accuracy trade-off.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Dashboard — post training
# ─────────────────────────────────────────────────────────────

p = st.session_state.pipeline
metrics = p.metrics
summary = metrics.get("summary", {})
naive_summary = metrics.get("naive_summary", {})
reconciled = p.reconciled
encoders = p.processed.get("encoders", {})

# ─── KPI Row ─────────────────────────────────────────────────
def pct_change(new, old):
    if old and old != 0:
        return (new - old) / abs(old) * 100
    return 0.0

kpi_cols = st.columns(5)
kpis = [
    ("MAE", summary.get("MAE", 0), naive_summary.get("MAE", 0), ".2f"),
    ("RMSE", summary.get("RMSE", 0), naive_summary.get("RMSE", 0), ".2f"),
    ("MAPE %", summary.get("MAPE", 0), naive_summary.get("MAPE", 0), ".1f"),
    ("SMAPE %", summary.get("SMAPE", 0), naive_summary.get("SMAPE", 0), ".1f"),
    ("Stores Modelled", len(p.prophet.models), 0, "d"),
]
kpi_colors = ["#0071CE", "#3FB950", "#FFC220", "#D29922", "#8B949E"]

for col, (label, val, naive_val, fmt), color in zip(kpi_cols, kpis, kpi_colors):
    with col:
        delta = pct_change(val, naive_val)
        if fmt == "d":
            val_str = str(int(val))
            delta_str = ""
        else:
            val_str = f"{val:{fmt}}"
            delta_str = f"{'▼' if delta < 0 else '▲'} {abs(delta):.1f}% vs naive"
        delta_class = "delta-neg" if delta < 0 else "delta-pos"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{val_str}</div>
            <div class="metric-label-text">{label}</div>
            <div class="metric-delta {delta_class}">{delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecasts", "🏪 Store Analysis", "🔗 Hierarchy", "⚡ Model Insights", "📐 Evaluation"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Forecasts
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><div class="section-dot"></div>Forecast vs Actuals</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([2, 2])
    with col_ctrl1:
        store_opts = list(encoders.get("store_id", {1: "Store 1"}).items())
        store_labels = [f"{v}" for k, v in store_opts]
        sel_store_label = st.selectbox("Store", store_labels, key="store_sel")
        sel_store_code = {v: k for k, v in encoders.get("store_id", {}).items()}.get(sel_store_label, 0)
    with col_ctrl2:
        agg_level = st.selectbox("Aggregation", ["Item (sample)", "Department", "Category", "Store Total"])

    actuals_df = metrics.get("lgbm_actuals")
    forecasts_df = metrics.get("lgbm_forecasts")

    if actuals_df is not None and forecasts_df is not None:
        if agg_level == "Store Total":
            # Aggregate all items for selected store
            store_actuals = actuals_df.merge(
                forecasts_df[["id", "date", "store_id"]], on=["id", "date"], how="left"
            )
            store_actuals = store_actuals[store_actuals["store_id"] == sel_store_code]
            agg_act = store_actuals.groupby("date")["sales"].sum().reset_index()
            agg_fcast = forecasts_df[forecasts_df["store_id"] == sel_store_code].groupby("date")["forecast"].sum().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg_act["date"], y=agg_act["sales"],
                                      name="Actual", line=dict(color=COLORS["actual"], width=2)))
            fig.add_trace(go.Scatter(x=agg_fcast["date"], y=agg_fcast["forecast"],
                                      name="LightGBM Forecast", line=dict(color=COLORS["lgbm"], width=2.5, dash="dash")))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Store {sel_store_label} — Total Sales", height=380)

        elif agg_level == "Department":
            store_fcast = forecasts_df[forecasts_df["store_id"] == sel_store_code]
            dept_fcast = store_fcast.groupby(["dept_id", "date"])["forecast"].sum().reset_index()
            store_act = actuals_df.merge(forecasts_df[["id", "store_id", "dept_id"]].drop_duplicates(),
                                          on="id", how="left")
            store_act = store_act[store_act["store_id"] == sel_store_code]
            dept_act = store_act.groupby(["dept_id", "date"])["sales"].sum().reset_index()

            dept_enc = encoders.get("dept_id", {})
            unique_depts = dept_fcast["dept_id"].unique()[:4]
            fig = make_subplots(rows=2, cols=2, subplot_titles=[
                dept_enc.get(d, str(d)) for d in unique_depts
            ])
            for idx, dept in enumerate(unique_depts):
                r, c = divmod(idx, 2)
                da = dept_act[dept_act["dept_id"] == dept]
                df_ = dept_fcast[dept_fcast["dept_id"] == dept]
                fig.add_trace(go.Scatter(x=da["date"], y=da["sales"], name="Actual",
                                          line=dict(color=COLORS["actual"], width=1.5),
                                          showlegend=(idx == 0)), row=r+1, col=c+1)
                fig.add_trace(go.Scatter(x=df_["date"], y=df_["forecast"], name="Forecast",
                                          line=dict(color=COLORS["lgbm"], width=2, dash="dash"),
                                          showlegend=(idx == 0)), row=r+1, col=c+1)
            fig.update_layout(**PLOTLY_LAYOUT, height=480, title="Department-Level Forecasts")

        else:  # Item sample
            store_ids = forecasts_df[forecasts_df["store_id"] == sel_store_code]["id"].unique()
            sample_id = store_ids[0] if len(store_ids) > 0 else None
            if sample_id:
                item_act = actuals_df[actuals_df["id"] == sample_id]
                item_fcast = forecasts_df[forecasts_df["id"] == sample_id]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=item_act["date"], y=item_act["sales"],
                                          name="Actual", line=dict(color=COLORS["actual"], width=2),
                                          fill="tozeroy", fillcolor="rgba(230,237,243,0.05)"))
                fig.add_trace(go.Scatter(x=item_fcast["date"], y=item_fcast["forecast"],
                                          name="LightGBM", line=dict(color=COLORS["lgbm"], width=2.5)))
                fig.update_layout(**PLOTLY_LAYOUT, title=f"Item: {sample_id}", height=360)
            else:
                fig = go.Figure()
                fig.update_layout(**PLOTLY_LAYOUT, height=360)

        st.plotly_chart(fig, use_container_width=True)

    # Prophet store-level forecasts
    st.markdown('<div class="section-header"><div class="section-dot"></div>Prophet Store-Level Decomposition</div>', unsafe_allow_html=True)
    prophet_keys = list(p.prophet.models.keys())
    if prophet_keys:
        sel_prophet_store = st.selectbox("Prophet store", prophet_keys, key="prophet_sel")
        if sel_prophet_store and sel_prophet_store in p.prophet.forecasts_:
            fcast_df = p.prophet.forecasts_[sel_prophet_store]
            if fcast_df is not None and len(fcast_df) > 0:
                tail = fcast_df.tail(400)
                fig_p = make_subplots(rows=2, cols=2,
                                       subplot_titles=["Forecast (yhat)", "Trend", "Weekly Seasonality", "Yearly Seasonality"])
                cols_map = [("yhat", COLORS["lgbm"], 1, 1), ("trend", COLORS["prophet"], 1, 2),
                            ("weekly", "#3FB950", 2, 1), ("yearly", "#D29922", 2, 2)]
                for col_name, color, row, col in cols_map:
                    if col_name in tail.columns:
                        fig_p.add_trace(go.Scatter(x=tail["ds"], y=tail[col_name],
                                                    name=col_name, line=dict(color=color, width=1.5),
                                                    showlegend=False), row=row, col=col)
                fig_p.update_layout(**PLOTLY_LAYOUT, height=500, title=f"Prophet: {sel_prophet_store}")
                st.plotly_chart(fig_p, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Store Analysis
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><div class="section-dot"></div>Store Performance Matrix</div>', unsafe_allow_html=True)

    forecasts_df = metrics.get("lgbm_forecasts")
    actuals_df = metrics.get("lgbm_actuals")
    merged_eval = metrics.get("merged_eval")

    if merged_eval is not None and len(merged_eval) > 0:
        # Per-store metrics
        store_enc = encoders.get("store_id", {})
        store_metrics = []
        store_col = "store_id_x" if "store_id_x" in merged_eval.columns else "store_id"
        for store_code in merged_eval[store_col].unique():
            sub = merged_eval[merged_eval[store_col] == store_code]
            a_ = sub["sales"].values
            f_ = sub["forecast"].values
            from src.evaluation.metrics import mae as _mae, rmse as _rmse, mape as _mape
            store_metrics.append({
                "Store": store_enc.get(store_code, str(store_code)),
                "MAE": _mae(a_, f_),
                "RMSE": _rmse(a_, f_),
                "MAPE%": _mape(a_, f_),
                "Total Sales": float(a_.sum()),
                "Total Forecast": float(f_.sum()),
            })
        sdf = pd.DataFrame(store_metrics)
        sdf["Forecast Bias%"] = (sdf["Total Forecast"] - sdf["Total Sales"]) / sdf["Total Sales"].clip(lower=1) * 100

        # Heatmap
        heat_data = sdf[["Store", "MAE", "RMSE", "MAPE%", "Forecast Bias%"]].set_index("Store")
        fig_heat = px.imshow(
            heat_data.T,
            color_continuous_scale=[[0, "#0D3B66"], [0.5, "#0071CE"], [1, "#FFC220"]],
            text_auto=".1f",
            aspect="auto",
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=280, title="Store × Metric Heatmap",
                                coloraxis_showscale=False)
        fig_heat.update_traces(textfont=dict(size=11, color="white"))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Bar: MAE per store
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=sdf["Store"], y=sdf["MAE"],
            marker_color=[COLORS["lgbm"]] * len(sdf),
            marker_line_color="#FFC220", marker_line_width=0.5,
            name="MAE",
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, title="MAE by Store", height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter: Sales vs Forecast
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=sdf["Total Sales"], y=sdf["Total Forecast"],
            mode="markers+text",
            text=sdf["Store"],
            textposition="top center",
            marker=dict(size=12, color=COLORS["lgbm"], line=dict(color=COLORS["prophet"], width=1.5)),
        ))
        max_val = max(sdf["Total Sales"].max(), sdf["Total Forecast"].max()) * 1.05
        fig_scat.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                       line=dict(color="#8B949E", dash="dot"), name="Perfect forecast"))
        fig_scat.update_layout(**PLOTLY_LAYOUT, title="Forecast vs Actual Total Sales (by Store)",
                                height=380, xaxis_title="Actual", yaxis_title="Forecast")
        st.plotly_chart(fig_scat, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Hierarchy
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><div class="section-dot"></div>Hierarchical Reconciliation</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
        <div class="info-box">
        <b>Bottom-Up Reconciliation</b><br><br>
        1. Train LightGBM on item×store level (bottom level)<br>
        2. Sum item-store forecasts → store → state → total<br>
        3. Coherence guaranteed by construction<br>
        4. No information lost; all granularity preserved<br>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-box">
        <b>Hierarchy Levels (M5)</b><br><br>
        • <b>Total</b>: 1 series<br>
        • <b>State</b>: 3 series (CA, TX, WI)<br>
        • <b>Store</b>: 10 series<br>
        • <b>Category</b>: 3 series<br>
        • <b>Department</b>: 7 series<br>
        • <b>Item</b>: {n_items * 7} series (sample)<br>
        • <b>Item×Store (bottom)</b>: {n_items * 7 * 10} series (sample)<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if reconciled:
        # Show reconciled totals at each level
        levels_to_show = {}

        if "total" in reconciled:
            tot = reconciled["total"]
            if "forecast" in tot.columns and "date" in tot.columns:
                levels_to_show["Total"] = tot.groupby("date")["forecast"].sum()

        if "store" in reconciled:
            st_data = reconciled["store"]
            if "forecast" in st_data.columns and "date" in st_data.columns:
                store_enc2 = encoders.get("store_id", {})
                # Show top 3 stores
                store_cols = [c for c in st_data.columns if c not in ["date", "forecast"]]
                if store_cols:
                    for store_code in st_data[store_cols[0]].unique()[:3]:
                        label = store_enc2.get(store_code, str(store_code))
                        sub = st_data[st_data[store_cols[0]] == store_code]
                        levels_to_show[f"Store {label}"] = sub.groupby("date")["forecast"].sum()

        if levels_to_show:
            fig_hier = go.Figure()
            level_colors = ["#0071CE", "#FFC220", "#3FB950", "#D29922", "#8B949E"]
            for (name, series), color in zip(levels_to_show.items(), level_colors):
                fig_hier.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    name=name, line=dict(color=color, width=2)
                ))
            fig_hier.update_layout(**PLOTLY_LAYOUT, title="Reconciled Forecasts by Level",
                                    height=400, yaxis_title="Units Sold (forecast)")
            st.plotly_chart(fig_hier, use_container_width=True)

        # Coherence check
        coherent = p.reconciler.check_coherence(reconciled)
        if coherent:
            st.markdown('<div class="info-box">✅ <b>Coherence check passed</b>: Store-level forecasts sum to total within tolerance.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ Coherence check failed — consider rerunning with bottom_up method.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — Model Insights
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><div class="section-dot"></div>LightGBM Feature Importance</div>', unsafe_allow_html=True)

    fi = p.lgbm.feature_importance_
    if fi is not None:
        top_n = 25
        fi_top = fi.head(top_n).sort_values("importance")

        # Color by feature type
        def feat_color(f):
            if "lag" in f: return "#0071CE"
            if "rolling" in f: return "#3FB950"
            if "price" in f: return "#FFC220"
            if any(x in f for x in ["sin", "cos", "day", "week", "month", "year", "quarter"]): return "#D29922"
            if any(x in f for x in ["store", "cat", "dept", "item", "state"]): return "#8B949E"
            return "#C0C0C0"

        colors = [feat_color(f) for f in fi_top["feature"]]

        fig_fi = go.Figure(go.Bar(
            x=fi_top["importance"],
            y=fi_top["feature"],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=600,
                      title=f"Top {top_n} Features by Gain",
                      xaxis_title="Importance (Gain)")
        fig_fi.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig_fi, use_container_width=True)

        # Legend
        c1, c2, c3, c4, c5 = st.columns(5)
        legend_items = [
            (c1, "#0071CE", "Lag features"),
            (c2, "#3FB950", "Rolling features"),
            (c3, "#FFC220", "Price features"),
            (c4, "#D29922", "Calendar features"),
            (c5, "#8B949E", "ID embeddings"),
        ]
        for col, color, label in legend_items:
            with col:
                st.markdown(f'<div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;color:#8B949E;"><div style="width:12px;height:12px;border-radius:2px;background:{color};"></div>{label}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-dot"></div>Model Configuration</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        lgbm_params = {k: v for k, v in p.lgbm.params.items()}
        st.markdown("""
        <div class="info-box">
        <b>LightGBM Hyperparameters</b><br><br>
        """ + "<br>".join([f"• <b>{k}</b>: {v}" for k, v in list(lgbm_params.items())[:8]]) + """
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b>Prophet Configuration</b><br><br>
        • <b>Mode</b>: Multiplicative seasonality<br>
        • <b>Yearly seasonality</b>: True<br>
        • <b>Weekly seasonality</b>: True<br>
        • <b>Changepoint prior</b>: 0.05<br>
        • <b>Holidays</b>: SuperBowl, Thanksgiving, Christmas, Easter, ValentinesDay<br>
        • <b>Regressors</b>: SNAP flags<br>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — Evaluation
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><div class="section-dot"></div>Evaluation Report</div>', unsafe_allow_html=True)

    summary = metrics.get("summary", {})
    naive_summary = metrics.get("naive_summary", {})

    # Comparison table
    if summary and naive_summary:
        comp_data = {
            "Metric": ["MAE", "RMSE", "MAPE %", "SMAPE %"],
            "LightGBM": [
                f"{summary.get('MAE', 0):.3f}",
                f"{summary.get('RMSE', 0):.3f}",
                f"{summary.get('MAPE', 0):.2f}",
                f"{summary.get('SMAPE', 0):.2f}",
            ],
            "Naive Baseline": [
                f"{naive_summary.get('MAE', 0):.3f}",
                f"{naive_summary.get('RMSE', 0):.3f}",
                f"{naive_summary.get('MAPE', 0):.2f}",
                f"{naive_summary.get('SMAPE', 0):.2f}",
            ],
            "Improvement": [
                f"{pct_change(summary.get('MAE', 1), naive_summary.get('MAE', 1)):+.1f}%",
                f"{pct_change(summary.get('RMSE', 1), naive_summary.get('RMSE', 1)):+.1f}%",
                f"{pct_change(summary.get('MAPE', 1), naive_summary.get('MAPE', 1)):+.1f}%",
                f"{pct_change(summary.get('SMAPE', 1), naive_summary.get('SMAPE', 1)):+.1f}%",
            ],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Residuals distribution
    merged_eval = metrics.get("merged_eval")
    if merged_eval is not None and len(merged_eval) > 0:
        residuals = merged_eval["sales"].values - merged_eval["forecast"].values
        fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Residuals Distribution", "Residuals Over Time"])

        fig_res.add_trace(go.Histogram(x=residuals, nbinsx=60,
                                        marker_color=COLORS["lgbm"], opacity=0.8,
                                        name="Residuals"), row=1, col=1)
        sample_dates = merged_eval.sort_values("date")["date"].values[:500]
        sample_resid = residuals[:500]
        fig_res.add_trace(go.Scatter(x=sample_dates, y=sample_resid, mode="markers",
                                      marker=dict(size=3, color=COLORS["lgbm"], opacity=0.5),
                                      name="Residuals"), row=1, col=2)
        fig_res.add_hline(y=0, line_color="#8B949E", line_dash="dot", row=1, col=2)
        fig_res.update_layout(**PLOTLY_LAYOUT, height=350, title="LightGBM Residual Analysis",
                               showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

    # WRMSSE explanation
    st.markdown('<div class="section-header"><div class="section-dot"></div>WRMSSE Metric Explanation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>WRMSSE = Weighted Root Mean Squared Scaled Error</b><br><br>
    • The official M5 competition metric<br>
    • <b>Scale</b>: divided by MSE of naive 1-step-ahead forecast (random walk) on training data<br>
    • <b>Weight</b>: each series weighted by its revenue contribution (price × volume) over last 28 training days<br>
    • <b>Aggregation</b>: evaluated at all 12 aggregation levels simultaneously<br>
    • <b>WRMSSE < 1.0</b> means the model beats the naive baseline<br>
    • M5 winner achieved WRMSSE ≈ 0.50 (50% improvement over naive)
    </div>
    """, unsafe_allow_html=True)

    # Data summary
    st.markdown('<div class="section-header"><div class="section-dot"></div>Dataset Summary</div>', unsafe_allow_html=True)
    raw_sales = p.raw_data.get("sales")
    if raw_sales is not None:
        n_items_total = len(raw_sales["item_id"].unique())
        n_series = len(raw_sales)
        n_days = len([c for c in raw_sales.columns if c.startswith("d_")])
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Unique Items", f"{n_items_total:,}"),
            (c2, "Item×Store Series", f"{n_series:,}"),
            (c3, "History Days", f"{n_days:,}"),
            (c4, "Forecast Horizon", f"{p.horizon} days"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.5rem; color:#0071CE;">{val}</div>
                    <div class="metric-label-text">{label}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top: 1px solid #30363D; padding-top: 1rem; text-align: center;">
    <span style="font-family: IBM Plex Mono, monospace; font-size: 0.78rem; color: #8B949E;">
        Walmart M5 Store Sales Forecasting · LightGBM + Prophet + Hierarchical Reconciliation · 
        Built for FANG-level ML Engineering Portfolio
    </span>
</div>
""", unsafe_allow_html=True)
