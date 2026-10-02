"""
Streamlit Web Application
EV Charging Station Congestion & Grid Load Predictor

A modern, high-polish dashboard for EV drivers and station operators to
forecast charging congestion, grid load, waiting times, and find nearby
alternative stations.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import get_predictor

# Page Configuration
st.set_page_config(
    page_title="EV Charging Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clean Modern UI
st.markdown(
    """
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .app-header {
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 50%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .card-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .card-footer {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.4rem;
    }
    
    /* Vibrant Status Badges */
    .badge-low {
        background: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 1.35rem;
        padding: 6px 20px;
        border-radius: 30px;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-medium {
        background: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        font-size: 1.35rem;
        padding: 6px 20px;
        border-radius: 30px;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .badge-high {
        background: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        font-size: 1.35rem;
        padding: 6px 20px;
        border-radius: 30px;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    
    /* Recommendation Banner */
    .recom-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 14px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.15);
    }
    .saved-badge {
        background: #10B981;
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Tier Cards */
    .tier-box {
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }
    .tier-low {
        background: #F0FDF4;
        border-color: #BBF7D0;
    }
    .tier-medium {
        background: #FFFBEB;
        border-color: #FDE68A;
    }
    .tier-high {
        background: #FEF2F2;
        border-color: #FECACA;
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown(
    """
    <div class="app-header">
        <h1 class="main-title">⚡ EV Charging Congestion & Grid Load Predictor</h1>
        <p class="sub-title">AI-powered station forecasting: Predict wait times, congestion levels, and discover less-crowded alternative chargers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Prediction Engine
@st.cache_resource(show_spinner=False)
def load_app_predictor():
    try:
        return get_predictor(), None
    except Exception as e:
        return None, str(e)

predictor, load_err = load_app_predictor()
if load_err:
    st.error(f"Failed to initialize prediction models: {load_err}")
    st.info("Please run `python src/train_models.py` in your terminal to train and serialize the models.")
    st.stop()

# Sidebar: Parameters
st.sidebar.markdown("### ⚙️ Search Parameters")
st.sidebar.caption("Select your target station and time to view AI predictions.")

stations = predictor.get_available_stations()
days = predictor.get_available_days()
hours = list(range(24))

def format_hour(h):
    period = "AM" if h < 12 else "PM"
    h12 = h if (h == 12 or h == 0) else (h % 12)
    h12_str = 12 if h12 == 0 else h12
    return f"{h:02d}:00 ({h12_str:02d}:00 {period})"

# Set sensible defaults
default_station = 129465 if 129465 in stations else stations[0]
default_idx = stations.index(default_station) if default_station in stations else 0

selected_station = st.sidebar.selectbox("Charging Station ID", stations, index=default_idx)
selected_day = st.sidebar.selectbox("Day of Week", days, index=4)  # Friday peak
selected_hour = st.sidebar.selectbox("Hour of Day", hours, index=17, format_func=format_hour)

run_button = st.sidebar.button("⚡ Forecast Station Availability", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Engine")
st.sidebar.caption(
    "Predictions are powered by **Logistic Regression**, **Decision Trees**, and **Random Forests** "
    "evaluating concurrent session frequency, average charge duration, and energy demand."
)

# Run Prediction
try:
    result = predictor.run_full_prediction(selected_station, selected_day, selected_hour)
except Exception as e:
    st.error(f"Inference error: {e}")
    st.info("Historical data is unavailable for this station/hour. Please select another station.")
    st.stop()

# Clean Navigation Tabs
tab_live, tab_compare, tab_tiers, tab_network = st.tabs([
    "⚡ Live Station Forecast",
    "📊 3-Model Comparison & 24h Trend",
    "ℹ Congestion Tiers Guide",
    "📈 Network Analytics & Accuracy"
])

# -------------------------------------------------------------
# TAB 1: LIVE FORECAST & SMART RECOMMENDATIONS
# -------------------------------------------------------------
with tab_live:
    if not result["exact_historical_match"]:
        st.info(
            f"ℹ No exact historical sessions found for Station {selected_station} at {format_hour(selected_hour)}. "
            "Forecast is calculated using the station's historical baseline profile."
        )

    # 1. Hero KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)

    cong_level = result["final_congestion"]
    grid_level = result["grid_load"]
    wait_time = result["estimated_wait_min"]

    badge_cls = f"badge-{cong_level.lower()}"
    grid_cls = f"badge-{grid_level.lower()}"

    desc_map = {
        "Low": "Optimal baseline power demand",
        "Medium": "Moderate grid demand",
        "High": "Peak localized demand stress",
    }

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Congestion Level</div>
                <div><span class="{badge_cls}">{cong_level}</span></div>
                <div class="card-footer">Consensus from 3 ML models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Grid Load Status</div>
                <div><span class="{grid_cls}">{grid_level}</span></div>
                <div class="card-footer">{desc_map.get(grid_level, '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Estimated Wait Time</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #1E293B;">{wait_time} <span style="font-size: 1.1rem; font-weight: 600; color: #64748B;">min</span></div>
                <div class="card-footer">Calculated from historical session turnover</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Smart Alternative Station Recommendation (When High)
    if cong_level == "High":
        recom = result["recommendation"]
        if recom:
            st.markdown(
                f"""
                <div class="recom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="margin: 0; color: #1E3A8A; font-weight: 700; font-size: 1.3rem;">
                            📍 Recommended Nearby Alternative: Station {recom['station_id']}
                        </h3>
                        <span class="saved-badge">⏱ Saves {recom['wait_time_saved_min']} min wait!</span>
                    </div>
                    <p style="font-size: 1.05rem; margin: 0.5rem 0; color: #1E293B;">
                        <strong>Proximity:</strong> {recom.get('proximity', 'Network Station')} &nbsp;|&nbsp;
                        <strong>Congestion:</strong> <span class="badge-low" style="font-size: 0.95rem; padding: 3px 12px;">{recom['congestion_level']}</span> &nbsp;|&nbsp;
                        <strong>Estimated Wait:</strong> <strong>{recom['estimated_wait_min']} min</strong>
                    </p>
                    <p style="color: #334155; margin-bottom: 0.4rem; font-size: 0.95rem;">
                        <strong>Why Recommended:</strong> {recom['reason']}
                    </p>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.5rem;">
                        Historical Activity at {format_hour(selected_hour)}: <strong>{recom['sessions_at_station_hour']} session(s)</strong> | 
                        Avg Energy: <strong>{round(recom['avg_energy_kwh'], 2)} kWh</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠ High congestion predicted, but no less-congested alternative station was available at this hour.")
    elif cong_level == "Medium":
        st.info(f"ℹ **Moderate Traffic**: Station {selected_station} has moderate occupancy. Expected wait is approximately **{wait_time} minutes**.")
    else:
        st.success(f"✅ **Open Plugs Available**: Station {selected_station} is predicted to have Low congestion with virtually no waiting time.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Individual Model Forecasts
    st.subheader("🤖 Individual Machine Learning Predictions")
    preds_table = []
    for m_name, pred_val in result["model_predictions"].items():
        conf = result["model_probabilities"][m_name][pred_val] * 100.0
        preds_table.append({
            "Machine Learning Model": m_name,
            "Predicted Congestion": pred_val,
            "Confidence Probability": f"{conf:.1f}%",
            "Consensus Agreement": "✅ Match" if pred_val == cong_level else "⚠ Divergent"
        })

    m_col1, m_col2 = st.columns([3, 2])
    with m_col1:
        st.dataframe(pd.DataFrame(preds_table), hide_index=True)
    with m_col2:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem; font-size: 0.9rem;">
                <strong>Consensus Decision Rule:</strong><br>
                The final congestion classification uses <strong>majority voting</strong> among all 3 models.
                If all three models predict differing classes, the <strong>Random Forest</strong> serves as the academic tie-breaker.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Historical Profile Cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"📊 Station {selected_station} Historical Profile ({format_hour(selected_hour)})")
    h1, h2, h3 = st.columns(3)
    h1.metric("Historical Session Volume", f"{result['historical_sessions']} sessions")
    h2.metric("Average Energy Delivered", f"{result['avg_energy_kwh']} kWh")
    h3.metric("Average Charging Duration", f"{result['avg_charge_time_min']} min")

# -------------------------------------------------------------
# TAB 2: 3-MODEL COMPARISON & 24H TRENDS
# -------------------------------------------------------------
with tab_compare:
    st.subheader(f"📈 3-Model Comparison & 24-Hour Trend for Station {selected_station}")
    st.markdown("Visualizing model agreement, class probability distributions, and diurnal shifts across the entire 24-hour cycle.")

    c_col1, c_col2 = st.columns(2)

    # Model Probabilities Bar Chart
    with c_col1:
        st.markdown(f"**Model Class Probabilities at {format_hour(selected_hour)}**")
        prob_rows = []
        for m_name, p_dict in result["model_probabilities"].items():
            for tier, prob in p_dict.items():
                prob_rows.append({"Model": m_name, "Tier": tier, "Probability": prob})
        prob_df = pd.DataFrame(prob_rows)

        fig_p, ax_p = plt.subplots(figsize=(6.5, 4))
        sns.barplot(
            data=prob_df,
            x="Model",
            y="Probability",
            hue="Tier",
            palette={"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"},
            ax=ax_p
        )
        ax_p.set_title(f"Prediction Probability Distribution", fontsize=11, fontweight="bold")
        ax_p.set_ylabel("Probability (0.0 to 1.0)")
        ax_p.set_ylim(0, 1.05)
        ax_p.legend(title="Tier", loc="upper right")
        st.pyplot(fig_p)

    # 24-Hour Model Comparison Line Chart
    with c_col2:
        st.markdown(f"**24-Hour Model Congestion Shifts ({selected_day})**")
        df_24h = predictor.get_24h_model_comparison(selected_station, selected_day)

        fig_24, ax_24 = plt.subplots(figsize=(6.5, 4))
        ax_24.plot(df_24h["hour"], df_24h["LR_Numeric"], linestyle="--", marker="o", markersize=3, label="Logistic Regression", color="#3B82F6", alpha=0.7)
        ax_24.plot(df_24h["hour"], df_24h["DT_Numeric"], linestyle=":", marker="s", markersize=3, label="Decision Tree", color="#10B981", alpha=0.7)
        ax_24.plot(df_24h["hour"], df_24h["RF_Numeric"], linestyle="-.", marker="^", markersize=3, label="Random Forest", color="#8B5CF6", alpha=0.7)
        ax_24.plot(df_24h["hour"], df_24h["Consensus_Numeric"], linewidth=2.5, marker="D", markersize=5, label="Final Consensus", color="#EF4444")
        
        ax_24.axvline(x=selected_hour, color="#64748B", linestyle="--", linewidth=1.5, label=f"Selected ({selected_hour}h)")
        ax_24.set_yticks([1, 2, 3])
        ax_24.set_yticklabels(["Low", "Medium", "High"])
        ax_24.set_xlabel("Hour (0 to 23)")
        ax_24.set_ylabel("Congestion Tier")
        ax_24.set_title("24h Predictions Across 3 Models", fontsize=11, fontweight="bold")
        ax_24.legend(loc="upper right", fontsize=8)
        ax_24.grid(True, alpha=0.3)
        st.pyplot(fig_24)

    st.markdown("---")
    # 24-Hour Waiting Time Curve
    st.markdown(f"**24-Hour Estimated Waiting Time Curve (Station {selected_station})**")
    fig_w, ax_w = plt.subplots(figsize=(12, 3.2))
    ax_w.plot(df_24h["hour"], df_24h["Estimated Wait (min)"], color="#F59E0B", marker="o", linewidth=2.2)
    ax_w.fill_between(df_24h["hour"], df_24h["Estimated Wait (min)"], color="#F59E0B", alpha=0.15)
    ax_w.axvline(x=selected_hour, color="#EF4444", linestyle="--", linewidth=2, label=f"Selected ({selected_hour}:00 - {wait_time} min)")
    ax_w.set_xlabel("Hour of the Day (0 to 23)")
    ax_w.set_ylabel("Estimated Wait (min)")
    ax_w.set_title(f"Hourly Queue & Waiting Time Forecast", fontsize=11, fontweight="bold")
    ax_w.set_xticks(range(0, 24))
    ax_w.legend()
    st.pyplot(fig_w)

# -------------------------------------------------------------
# TAB 3: CONGESTION TIERS GUIDE
# -------------------------------------------------------------
with tab_tiers:
    st.subheader("ℹ Congestion Tiers Guide: Understanding Low, Medium & High")
    st.markdown("Definitions, calculated wait times, and recommended driver actions for each congestion status:")

    t_col1, t_col2, t_col3 = st.columns(3)

    with t_col1:
        st.markdown(
            """
            <div class="tier-box tier-low">
                <h3 style="color: #15803D; margin-top: 0;">🟢 Low Congestion</h3>
                <p><strong>Score Threshold:</strong> &le; 33rd Percentile</p>
                <p><strong>Turnover Status:</strong> Plugs are readily available; low simultaneous session volume.</p>
                <p><strong>Waiting Time Rule:</strong> <code>20% of Avg Duration</code></p>
                <p><strong>Expected Wait:</strong> 0 – 35 minutes</p>
                <p><strong>Driver Advice:</strong> Excellent time to charge. Immediate hookup expected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            """
            <div class="tier-box tier-medium">
                <h3 style="color: #B45309; margin-top: 0;">🟡 Medium Congestion</h3>
                <p><strong>Score Threshold:</strong> 33rd – 66th Percentile</p>
                <p><strong>Turnover Status:</strong> Moderate turnover; periodic slot occupancy.</p>
                <p><strong>Waiting Time Rule:</strong> <code>50% of Avg Duration</code></p>
                <p><strong>Expected Wait:</strong> 35 – 90 minutes</p>
                <p><strong>Driver Advice:</strong> Reasonable charging window. Brief queue possible during shift changes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col3:
        st.markdown(
            """
            <div class="tier-box tier-high">
                <h3 style="color: #B91C1C; margin-top: 0;">🔴 High Congestion</h3>
                <p><strong>Score Threshold:</strong> &gt; 66th Percentile</p>
                <p><strong>Turnover Status:</strong> Dense clustering of long charging sessions; high wait times.</p>
                <p><strong>Waiting Time Rule:</strong> <code>85% of Avg Duration</code></p>
                <p><strong>Expected Wait:</strong> 90 – 200+ minutes</p>
                <p><strong>Driver Advice:</strong> Severe delays likely. Reroute to recommended nearby charger.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# TAB 4: NETWORK ANALYTICS & ACCURACY
# -------------------------------------------------------------
with tab_network:
    st.subheader("🔬 Machine Learning Accuracy Benchmark")
    st.markdown("Performance metrics evaluated on the stratified 20% hold-out test set (400 samples):")

    metrics_file = PROJECT_ROOT / "models" / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        table_rows = []
        for m_name, vals in metrics_data.items():
            table_rows.append({
                "Model": m_name,
                "Accuracy": f"{vals['Accuracy']:.4f}",
                "Precision": f"{vals['Precision']:.4f}",
                "Recall": f"{vals['Recall']:.4f}",
                "F1 Score": f"{vals['F1 Score']:.4f}",
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True)

        st.markdown("### Confusion Matrices")
        cm1, cm2, cm3 = st.columns(3)
        labels = ["Low", "Medium", "High"]
        m_list = ["Logistic Regression", "Decision Tree", "Random Forest"]

        for k, c in zip(m_list, [cm1, cm2, cm3]):
            if k in metrics_data:
                with c:
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
                    cm = np.array(metrics_data[k]["Confusion Matrix"])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=labels, yticklabels=labels, ax=ax_cm)
                    ax_cm.set_title(f"{k}", fontsize=10, fontweight="bold")
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)
    else:
        st.info("Benchmark metrics not found. Run `python src/train_models.py` to regenerate.")
