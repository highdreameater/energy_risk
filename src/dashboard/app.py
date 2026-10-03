import os
import sys
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_raw_data, load_portfolio_exposures
from src.features.ratios import compute_commodity_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import (
    calculate_gini,
    calculate_ks_stat,
    calculate_brier_score,
    calculate_psi,
    compute_roc_curve_data,
    compute_ks_curve_data,
    compute_calibration_curve_data,
    generate_validation_report
)
from src.risk.rating_engine import assign_rating_bands, get_rating_scale
from src.risk.el_calculator import compute_expected_loss, aggregate_expected_loss
from src.risk.limit_monitor import evaluate_portfolio_limits, summarize_limit_monitoring
from src.risk.stress_tester import (
    run_portfolio_stress_test,
    run_counterparty_stress_simulation,
    compute_rating_migration_matrix,
    load_stress_scenarios
)

# Page configuration
st.set_page_config(
    page_title="ECCRE | Enterprise Credit Risk & Capital Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom institutional styling
st.markdown("""
<style>
    .reportview-container {
        background: #0B0F19;
    }
    .metric-card {
        background: linear-gradient(135deg, #131B2E 0%, #1A233A 100%);
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #233152;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 2px;
    }
    .status-badge-breach {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-warning {
        background-color: #78350F;
        color: #FCD34D;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-normal {
        background-color: #064E3B;
        color: #6EE7B7;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .header-box {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 5px solid #38BDF8;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_or_run_pipeline():
    processed_path = "data/processed/model_predictions.parquet"
    if os.path.exists(processed_path):
        df_final = pd.read_parquet(processed_path)
    else:
        # Run pipeline
        raw_path = "data/raw/counterparty_financials.csv"
        if not os.path.exists(raw_path):
            from src.data.generate_mock_data import generate_market_data, generate_counterparty_data
            generate_market_data()
            generate_counterparty_data()
        df_raw = load_raw_data(raw_path)
        df_features = compute_commodity_ratios(df_raw)
        df_scored, _ = train_and_calibrate_pd_model(df_features, save_path=None)
        df_rated = assign_rating_bands(df_scored, config_path="config/rating_bands.yaml")
        df_el = compute_expected_loss(df_rated, exposures_path="data/portfolio/exposures_limits.csv")
        df_final = evaluate_portfolio_limits(df_el)
        os.makedirs("data/processed", exist_ok=True)
        df_final.to_parquet(processed_path, index=False)
    return df_final

# Load Data
df = get_or_run_pipeline()

# Sidebar Navigation & Global Filters
st.sidebar.image("https://img.icons8.com/isometric/100/38bdf8/energy-meter.png", width=60)
st.sidebar.title("ECCRE Risk Engine")
st.sidebar.markdown("**Enterprise Commodities & Energy Credit Risk & Capital Console**")
st.sidebar.markdown("---")

# Global Portfolio Filters
all_sectors = ["All Sectors"] + sorted(list(df["sector"].unique()))
selected_sector = st.sidebar.selectbox("Filter Sector", all_sectors)

all_ratings = ["All Ratings"] + ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
selected_rating = st.sidebar.selectbox("Filter Rating Band", all_ratings)

all_statuses = ["All Statuses", "Normal", "Warning", "BREACH"]
selected_status = st.sidebar.selectbox("Filter Limit Status", all_statuses)

st.sidebar.markdown("---")
st.sidebar.caption("Nomura Wholesale Risk Compliance Architecture")
st.sidebar.caption("Version: 1.0.0 (Production Release)")

# Apply Filters
filtered_df = df.copy()
if selected_sector != "All Sectors":
    filtered_df = filtered_df[filtered_df["sector"] == selected_sector]
if selected_rating != "All Ratings":
    filtered_df = filtered_df[filtered_df["rating_band"] == selected_rating]
if selected_status != "All Statuses":
    filtered_df = filtered_df[filtered_df["status"] == selected_status]

# Header Banner
st.markdown("""
<div class="header-box">
    <h2 style="margin: 0; color: #F8FAFC;">Enterprise Commodities & Energy Credit Risk & Capital Engine (ECCRE)</h2>
    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
        Institutional wholesale credit scoring, 252-day volatility feature engineering, Platt-calibrated PDs, and macro stress capital analytics.
    </p>
</div>
""", unsafe_allow_html=True)

# 4 Operational Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Executive Overview",
    "🎯 Model Validation Console (RMG)",
    "⚠️ Credit Exposure & Limit Monitoring (CEM)",
    "🌪️ Macro Stress-Testing Console"
])

# -------------------------------------------------------------------------------------------------
# TAB 1: PORTFOLIO EXECUTIVE OVERVIEW
# -------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("Wholesale Energy Portfolio Health & Capital Allocation")
    
    total_ead = filtered_df["ead"].sum()
    total_el = filtered_df["expected_loss"].sum()
    total_limit = filtered_df["credit_limit"].sum()
    portfolio_loss_rate = (total_el / max(total_ead, 1e-5)) * 10000.0
    breach_count = (filtered_df["status"] == "BREACH").sum()
    warning_count = (filtered_df["status"] == "Warning").sum()
    weighted_pd = (filtered_df["pd_calibrated"] * filtered_df["ead"]).sum() / max(total_ead, 1e-5) * 100.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Portfolio EAD</div>
            <div class="metric-value">${total_ead:,.1f}M</div>
            <div class="metric-delta" style="color: #38BDF8;">Limit: ${total_limit:,.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Expected Loss (EL)</div>
            <div class="metric-value" style="color: #F87171;">${total_el:,.2f}M</div>
            <div class="metric-delta" style="color: #F87171;">Loss Rate: {portfolio_loss_rate:.1f} bps</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Weighted 1-Yr PD</div>
            <div class="metric-value" style="color: #FBBF24;">{weighted_pd:.2f}%</div>
            <div class="metric-delta" style="color: #94A3B8;">Platt Calibrated</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Limit Breaches (>100%)</div>
            <div class="metric-value" style="color: {'#EF4444' if breach_count > 0 else '#10B981'};">{breach_count}</div>
            <div class="metric-delta" style="color: #EF4444;">Action Required</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Warnings (85-100%)</div>
            <div class="metric-value" style="color: #F59E0B;">{warning_count}</div>
            <div class="metric-delta" style="color: #F59E0B;">Watchlist Tier</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Portfolio Visualizations
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("##### Exposure (EAD) & Expected Loss by Energy Sector")
        sector_agg = aggregate_expected_loss(filtered_df, "sector")
        fig_sec = go.Figure()
        fig_sec.add_trace(go.Bar(
            x=sector_agg["sector"],
            y=sector_agg["total_ead"],
            name="EAD ($M)",
            marker_color="#38BDF8"
        ))
        fig_sec.add_trace(go.Bar(
            x=sector_agg["sector"],
            y=sector_agg["total_expected_loss"] * 10, # Scaled for visibility
            name="Expected Loss ($M x10)",
            marker_color="#F87171"
        ))
        fig_sec.update_layout(
            barmode="group",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with chart_col2:
        st.markdown("##### Internal Rating Distribution & Investment Grade Split")
        rating_counts = filtered_df["rating_band"].value_counts().reindex(["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]).fillna(0)
        colors = ["#10B981", "#34D399", "#60A5FA", "#818CF8", "#FBBF24", "#FB923C", "#EF4444"]
        fig_rating = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={"x": "Rating Band", "y": "Counterparties"},
            color=rating_counts.index,
            color_discrete_sequence=colors
        )
        fig_rating.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        
    # Interactive Counterparty Table
    st.markdown("##### Active Counterparty Master Risk Grid")
    display_cols = [
        "counterparty_id", "counterparty_name", "sector", "rating_band",
        "rating_tier", "pd_calibrated", "ead", "credit_limit", "utilization_pct",
        "expected_loss", "status"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="expected_loss", ascending=False),
        column_config={
            "pd_calibrated": st.column_config.NumberColumn("1-Yr Calibrated PD", format="%.4f"),
            "ead": st.column_config.NumberColumn("EAD ($M)", format="$%.1fM"),
            "credit_limit": st.column_config.NumberColumn("Limit ($M)", format="$%.1fM"),
            "utilization_pct": st.column_config.ProgressColumn("Utilization", format="%.1f%%", min_value=0, max_value=150),
            "expected_loss": st.column_config.NumberColumn("Expected Loss ($M)", format="$%.3fM"),
        },
        use_container_width=True,
        height=320
    )

# -------------------------------------------------------------------------------------------------
# TAB 2: MODEL VALIDATION CONSOLE (RMG)
# -------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("Risk Methodologies Group (RMG) Model Validation Suite")
    st.caption("Quantitative discrimination, calibration, and population stability assessment.")
    
    y_true = df["target"].values
    y_prob = df["pd_calibrated"].values
    
    # RMG Metric Scorecard Table
    report_df = generate_validation_report(y_true, y_prob)
    st.markdown("##### Model Governance Performance Scorecard")
    st.dataframe(
        report_df,
        column_config={
            "Model Value": st.column_config.NumberColumn("Computed Value", format="%.4f"),
            "Status": st.column_config.TextColumn("Status")
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    # 1. ROC Curve
    with v_col1:
        st.markdown("##### ROC Discrimination Curve")
        fpr, tpr, roc_auc = compute_roc_curve_data(y_true, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Scorecard (AUC = {roc_auc:.3f})', line=dict(color='#38BDF8', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='#64748B', dash='dash')))
        fig_roc.update_layout(
            template="plotly_dark",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    # 2. Cumulative KS Separation Chart
    with v_col2:
        st.markdown("##### Kolmogorov-Smirnov (KS) Separation")
        ks_df = compute_ks_curve_data(y_true, y_prob)
        ks_val = calculate_ks_stat(y_true, y_prob)
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=ks_df["cum_pop_pct"], y=ks_df["cum_default_pct"], mode='lines', name='Cum Defaults', line=dict(color='#EF4444', width=2.5)))
        fig_ks.add_trace(go.Scatter(x=ks_df["cum_pop_pct"], y=ks_df["cum_non_default_pct"], mode='lines', name='Cum Non-Defaults', line=dict(color='#10B981', width=2.5)))
        fig_ks.update_layout(
            template="plotly_dark",
            xaxis_title="Population Percentile (%)",
            yaxis_title="Cumulative Rate (%)",
            title=f"Max KS = {ks_val:.2f}%",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ks, use_container_width=True)
        
    # 3. Calibration Curve
    with v_col3:
        st.markdown("##### Platt Calibration Reliability")
        calib_df = compute_calibration_curve_data(y_true, y_prob)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=calib_df["pred_pd"], y=calib_df["obs_default_rate"], mode='markers+lines', name='Calibrated Model', marker=dict(color='#FBBF24', size=8)))
        fig_cal.add_trace(go.Scatter(x=[0, max(y_prob)], y=[0, max(y_prob)], mode='lines', name='Perfect Alignment', line=dict(color='#64748B', dash='dash')))
        fig_cal.update_layout(
            template="plotly_dark",
            xaxis_title="Mean Predicted PD",
            yaxis_title="Observed Default Rate",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cal, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 3: CREDIT EXPOSURE & LIMIT MONITORING (CEM)
# -------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("Credit Risk Exposure Management (CEM) Control Center")
    st.caption("Active limit utilization, collateral margin sensitivity, and breach investigations.")
    
    summary = summarize_limit_monitoring(df)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Sanctioned Limit", f"${df['credit_limit'].sum():,.1f}M")
    with c2:
        st.metric("Total Active Breaches", f"{summary['breach_count']} Counterparties", delta=f"${summary['total_excess_exposure']:,.1f}M Excess", delta_color="inverse")
    with c3:
        st.metric("Average Portfolio Utilization", f"{summary['avg_utilization_pct']:.1f}%")
    with c4:
        st.metric("Max Counterparty Utilization", f"{summary['max_utilization_pct']:.1f}%")
        
    st.markdown("---")
    
    cem_col1, cem_col2 = st.columns([1, 1])
    with cem_col1:
        st.markdown("##### Sector Exposure & Concentration Heatmap")
        fig_treemap = px.treemap(
            filtered_df,
            path=["sector", "rating_band", "counterparty_name"],
            values="ead",
            color="utilization_pct",
            color_continuous_scale="RdYlGn_r",
            color_continuous_midpoint=85.0
        )
        fig_treemap.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    with cem_col2:
        st.markdown("##### Limit Breach & Warning Watchlist")
        breach_warning_df = df[df["status"].isin(["BREACH", "Warning"])].copy()
        breach_warning_df = breach_warning_df.sort_values(by="utilization_pct", ascending=False)
        
        st.dataframe(
            breach_warning_df[[
                "counterparty_id", "counterparty_name", "sector", "rating_band",
                "ead", "credit_limit", "utilization_pct", "excess_exposure", "status"
            ]],
            column_config={
                "ead": st.column_config.NumberColumn("Exposure ($M)", format="$%.1fM"),
                "credit_limit": st.column_config.NumberColumn("Limit ($M)", format="$%.1fM"),
                "excess_exposure": st.column_config.NumberColumn("Excess ($M)", format="$%.2fM"),
                "utilization_pct": st.column_config.ProgressColumn("Utilization", format="%.1f%%", min_value=0, max_value=160),
            },
            use_container_width=True,
            height=420
        )

# -------------------------------------------------------------------------------------------------
# TAB 4: MACRO STRESS-TESTING CONSOLE
# -------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("Macroeconomic & Commodity Stress-Testing Simulation Console")
    st.caption("RMG Mandate: Simulating severe commodity price crashes, natural gas liquidity crunches, and stagflation shocks.")
    
    scenarios_dict = load_stress_scenarios("config/stress_scenarios.yaml")
    scenario_keys = list(scenarios_dict.keys())
    scenario_names = [scenarios_dict[k].get("name", k) for k in scenario_keys]
    
    st_col1, st_col2 = st.columns([1, 2])
    with st_col1:
        selected_scenario_name = st.selectbox("Select Macroeconomic Scenario", scenario_names)
        scenario_idx = scenario_names.index(selected_scenario_name)
        selected_key = scenario_keys[scenario_idx]
        spec = scenarios_dict[selected_key]
        
        st.info(f"**Scenario Description:** {spec.get('description', 'Severe macro scenario')}")
        st.markdown(f"**LGD Shift:** `+{spec.get('lgd_shift', 0.10)*100:.0f}%` across all sectors")
        st.markdown("**Sector PD Multipliers:**")
        mult_df = pd.DataFrame(list(spec.get("pd_multipliers", {}).items()), columns=["Sector", "PD Shock Multiplier"])
        st.dataframe(mult_df, hide_index=True, use_container_width=True)
        
    with st_col2:
        # Run Stress Test on entire portfolio
        stress_summary = run_portfolio_stress_test(df, "config/stress_scenarios.yaml")
        
        # Capital Impact Chart
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            x=stress_summary["scenario"],
            y=stress_summary["portfolio_el"],
            marker_color=["#10B981"] + ["#EF4444"] * (len(stress_summary) - 1),
            text=[f"${v:,.1f}M" for v in stress_summary["portfolio_el"]],
            textposition="outside"
        ))
        fig_stress.update_layout(
            template="plotly_dark",
            title="Portfolio Expected Loss (EL) Under Stress ($ Millions)",
            yaxis_title="Expected Loss ($M)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_stress, use_container_width=True)
        
    st.markdown("---")
    
    # Counterparty Stressed Simulation & Migration Matrix
    st.markdown(f"##### Rating Migration Matrix: Baseline Rating vs. Stressed Rating ({selected_scenario_name})")
    stressed_cp_df = run_counterparty_stress_simulation(
        df,
        scenario_key=selected_key,
        scenario_config_path="config/stress_scenarios.yaml",
        rating_config_path="config/rating_bands.yaml"
    )
    
    migration_matrix = compute_rating_migration_matrix(stressed_cp_df)
    
    mig_col1, mig_col2 = st.columns([1, 1])
    with mig_col1:
        fig_heatmap = px.imshow(
            migration_matrix,
            labels=dict(x="Stressed Rating", y="Baseline Rating", color="Counterparties"),
            x=migration_matrix.columns,
            y=migration_matrix.index,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_heatmap.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with mig_col2:
        downgraded = (stressed_cp_df["downgrade_notches"] > 0).sum()
        total_delta_el = stressed_cp_df["delta_expected_loss"].sum()
        pct_increase = (total_delta_el / df["expected_loss"].sum()) * 100.0
        
        st.markdown("###### Stress Scenario Summary Metrics")
        st.markdown(f"""
        - **Total Capital Depletion (Delta EL):** <span style="color:#EF4444; font-weight:bold;">+${total_delta_el:,.2f}M (+{pct_increase:.1f}%)</span>
        - **Counterparties Downgraded:** <span style="color:#F59E0B; font-weight:bold;">{downgraded} of {len(df)} ({downgraded/len(df)*100:.1f}%)</span>
        - **Post-Stress Expected Loss:** **${stressed_cp_df['stressed_expected_loss'].sum():,.2f}M**
        """, unsafe_allow_html=True)
        
        st.markdown("###### Top Capital Deteriorations")
        st.dataframe(
            stressed_cp_df.sort_values(by="delta_expected_loss", ascending=False)[[
                "counterparty_name", "sector", "rating_band", "stressed_rating_band",
                "pd_calibrated", "pd_stressed", "delta_expected_loss"
            ]].head(6),
            column_config={
                "pd_calibrated": st.column_config.NumberColumn("Base PD", format="%.3f"),
                "pd_stressed": st.column_config.NumberColumn("Stressed PD", format="%.3f"),
                "delta_expected_loss": st.column_config.NumberColumn("Δ EL ($M)", format="+$%.2fM")
            },
            hide_index=True,
            use_container_width=True,
            height=240
        )
