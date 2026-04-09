# app.py
# Drelviq — Autonomous Business Intelligence Platform
# Professional dark UI, smart routing, comparison, memory, email reports

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from agent import build_bi_agent, detect_columns, answer_with_memory

st.set_page_config(
    page_title="Drelviq — Autonomous BI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CACHE AGENT ───────────────────────────────────────────────────────────────


@st.cache_resource
def get_agent():
    return build_bi_agent()


# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main { background-color: #0F0F1A; }

.hero-header {
    background: linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle,
        rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-logo {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #6C63FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero-tagline {
    font-size: 1rem;
    color: #6C63FF;
    font-weight: 500;
    margin-top: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-subtitle {
    color: #8B8FA8;
    font-size: 0.95rem;
    margin-top: 12px;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}

.hero-badges {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #A09BFF;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E0E0E0;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg,
        rgba(108,99,255,0.4), transparent);
    margin-left: 8px;
}

.kpi-card {
    background: linear-gradient(135deg, #1A1A2E, #16213E);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}

.kpi-card:hover { border-color: rgba(108,99,255,0.5); }

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #6C63FF;
    line-height: 1;
}

.kpi-label {
    font-size: 0.8rem;
    color: #8B8FA8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.anomaly-spike {
    background: rgba(244,67,54,0.1);
    border-left: 3px solid #F44336;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.875rem;
    color: #EF9A9A;
}

.anomaly-drop {
    background: rgba(255,152,0,0.1);
    border-left: 3px solid #FF9800;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.875rem;
    color: #FFCC80;
}

.anomaly-ok {
    background: rgba(76,175,80,0.1);
    border-left: 3px solid #4CAF50;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.875rem;
    color: #A5D6A7;
}

.report-container {
    background: #1A1A2E;
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 12px;
    padding: 24px;
    line-height: 1.7;
}

.email-card {
    background: #1A1A2E;
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 20px;
}

.sidebar-node {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.sidebar-icon { font-size: 1rem; width: 24px; flex-shrink: 0; }
.sidebar-text { flex: 1; }
.sidebar-name { font-size: 0.8rem; font-weight: 600; color: #E0E0E0; }
.sidebar-desc { font-size: 0.7rem; color: #666; margin-top: 2px; }

.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(108,99,255,0.3), transparent);
    margin: 24px 0;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F0F1A; }
::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(26,26,46,0.8)",
    plot_bgcolor="rgba(26,26,46,0.0)",
    font=dict(color="#E0E0E0", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)"
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)"
    ),
    margin=dict(t=50, b=40, l=40, r=20)
)

COLORS = {
    "primary": "#6C63FF",
    "secondary": "#00D4AA",
    "warning": "#FF9800",
    "danger": "#F44336",
    "success": "#4CAF50",
    "chart_seq": ["#6C63FF", "#00D4AA", "#FF9800", "#F44336", "#E91E63"]
}

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-logo">⚡ Drelviq</div>
    <div class="hero-tagline">Autonomous Business Intelligence</div>
    <p class="hero-subtitle">
        Upload any dataset — Drelviq autonomously analyzes patterns,
        forecasts trends, detects anomalies, and generates grounded
        insights. Every claim backed by real data. Zero hallucination.
    </p>
    <div class="hero-badges">
        <span class="badge">⚡ LangGraph 10-Node Pipeline</span>
        <span class="badge">🧠 Smart Router</span>
        <span class="badge">🔮 3-Month Forecast</span>
        <span class="badge">🔀 Dataset Comparison</span>
        <span class="badge">💬 Conversational Memory</span>
        <span class="badge">🚫 Zero Hallucination</span>
        <span class="badge">📧 Email Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px 0;'>
        <div style='font-size:1.8rem;font-weight:800;
                    background:linear-gradient(135deg,#fff,#6C63FF);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    letter-spacing:-1px;'>
            ⚡ Drelviq
        </div>
        <div style='font-size:0.7rem;color:#555;
                    margin-top:4px;letter-spacing:2px;
                    text-transform:uppercase;'>
            10-node pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    nodes = [
        ("🔍", "Column Detection", "Auto-detects col types"),
        ("🧠", "Smart Router", "Decides analysis plan"),
        ("📋", "Data Summary", "KPIs and statistics"),
        ("📊", "Column Analysis", "Category breakdowns"),
        ("📈", "Trend Analysis", "Monthly / quarterly"),
        ("🔗", "Correlation", "Column relationships"),
        ("🔮", "Forecast", "Next 3 months"),
        ("🚨", "Anomaly Detection", "Spikes and drops"),
        ("🔀", "Comparison", "File 1 vs File 2"),
        ("🤖", "Report Gen", "Grounded AI report"),
    ]

    for icon, name, desc in nodes:
        st.markdown(f"""
        <div class="sidebar-node">
            <div class="sidebar-icon">{icon}</div>
            <div class="sidebar-text">
                <div class="sidebar-name">{name}</div>
                <div class="sidebar-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#555;line-height:1.8;'>
        <div>🟣 LLM: llama-3.1-8b-instant</div>
        <div>🌡️ Temperature: 0 (deterministic)</div>
        <div>📦 Framework: LangGraph</div>
        <div>📉 Forecast: LinearRegression</div>
        <div>🗂️ Charts: Plotly</div>
        <div>📧 Email: Resend API</div>
    </div>
    """, unsafe_allow_html=True)

# ── MODE SELECTOR ─────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>⚙️ Analysis Mode</p>",
            unsafe_allow_html=True)

mode = st.radio(
    "",
    ["Single Dataset Analysis", "Compare Two Datasets"],
    horizontal=True,
    label_visibility="collapsed"
)

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>📁 Data Upload</p>",
            unsafe_allow_html=True)

if mode == "Single Dataset Analysis":
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload any CSV file",
            type=["csv"],
            key="file1",
            help="Startups, Sales, HR, Finance — any structured CSV"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button(
            "📂 Load Sample",
            use_container_width=True
        )

    if use_sample:
        st.session_state["df1"] = pd.read_csv("startups_data.csv")
        st.session_state["df2"] = None
        st.success(
            "✅ Indian Startup Ecosystem dataset loaded "
            "— 500 rows, 9 columns"
        )

    if uploaded_file:
        st.session_state["df1"] = pd.read_csv(uploaded_file)
        st.session_state["df2"] = None
        st.success(
            f"✅ {uploaded_file.name} loaded — "
            f"{len(st.session_state['df1']):,} rows"
        )

else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='font-size:0.8rem;color:#8B8FA8;font-weight:600;
                    margin-bottom:8px;text-transform:uppercase;
                    letter-spacing:0.5px;'>
            📁 File 1 — Baseline (2024)
        </div>
        """, unsafe_allow_html=True)
        uploaded_file1 = st.file_uploader(
            "Upload File 1", type=["csv"], key="file1_compare"
        )
        if st.button("Load 2024 Startups", use_container_width=True):
            st.session_state["df1"] = pd.read_csv("startups_2024.csv")
            st.success("✅ File 1 loaded — startups_2024.csv")
        if uploaded_file1:
            st.session_state["df1"] = pd.read_csv(uploaded_file1)
            st.success(
                f"✅ File 1: {uploaded_file1.name} — "
                f"{len(st.session_state['df1']):,} rows"
            )

    with col2:
        st.markdown("""
        <div style='font-size:0.8rem;color:#8B8FA8;font-weight:600;
                    margin-bottom:8px;text-transform:uppercase;
                    letter-spacing:0.5px;'>
            📁 File 2 — Comparison (2025)
        </div>
        """, unsafe_allow_html=True)
        uploaded_file2 = st.file_uploader(
            "Upload File 2", type=["csv"], key="file2_compare"
        )
        if st.button("Load 2025 Startups", use_container_width=True):
            st.session_state["df2"] = pd.read_csv("startups_2025.csv")
            st.success("✅ File 2 loaded — startups_2025.csv")
        if uploaded_file2:
            st.session_state["df2"] = pd.read_csv(uploaded_file2)
            st.success(
                f"✅ File 2: {uploaded_file2.name} — "
                f"{len(st.session_state['df2']):,} rows"
            )

# ── DATA PREVIEW ──────────────────────────────────────────────────────────────
df1 = st.session_state.get("df1")
df2 = st.session_state.get("df2")

if df1 is not None:
    with st.expander("👁️ Preview Data", expanded=False):
        if df2 is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**File 1**")
                st.dataframe(df1.head(10), use_container_width=True)
            with c2:
                st.markdown("**File 2**")
                st.dataframe(df2.head(10), use_container_width=True)
        else:
            st.dataframe(df1.head(20), use_container_width=True)

        col_types = detect_columns(df1)
        c1, c2, c3 = st.columns(3)
        c1.info(f"📊 Numeric: {col_types['numeric']}")
        c2.info(f"🏷️ Categorical: {col_types['categorical']}")
        c3.info(f"📅 Date: {col_types['date']}")

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # ── RUN BUTTON ─────────────────────────────────────────────────────────
    run_col1, run_col2 = st.columns([4, 1])
    with run_col1:
        run_btn = st.button(
            "⚡ Run Drelviq Analysis",
            type="primary",
            use_container_width=True
        )
    with run_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        for key in ["result", "conversation_history", "all_data_context"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if run_btn:
        comparison_mode = (
            mode == "Compare Two Datasets" and df2 is not None
        )

        progress_bar = st.progress(0)
        status = st.empty()

        step_labels = [
            "🔍 Detecting column types...",
            "🧠 Smart router planning analysis...",
            "📋 Computing data summary...",
            "📊 Analyzing categories...",
            "📈 Detecting trends...",
            "🔗 Computing correlations...",
            "🔮 Forecasting next 3 months...",
            "🚨 Detecting anomalies...",
            "🔀 Comparing datasets..." if comparison_mode
            else "🔀 Skipping comparison...",
            "🤖 Generating AI report..."
        ]

        agent = get_agent()
        initial_state = {
            "query": "Generate a full business intelligence report",
            "df_json": df1.to_json(),
            "df2_json": df2.to_json() if df2 is not None else "",
            "comparison_mode": comparison_mode,
            "columns": df1.columns.tolist(),
            "numeric_cols": [],
            "categorical_cols": [],
            "date_cols": [],
            "run_trend": False,
            "run_forecast": False,
            "run_correlation": False,
            "run_comparison": False,
            "run_anomaly": False,
            "router_reasoning": "",
            "data_summary": "",
            "column_analysis": "",
            "correlation_analysis": "",
            "trend_analysis": "",
            "forecast_analysis": "",
            "anomalies": "",
            "comparison_analysis": "",
            "insights": [],
            "final_report": "",
            "chart_data": {},
            "nl_answer": "",
            "conversation_history": []
        }

        for i, label in enumerate(step_labels):
            status.markdown(
                f"<div style='color:#8B8FA8;font-size:0.85rem;'>"
                f"{label}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.2)
            progress_bar.progress((i + 1) / len(step_labels))

        result = agent.invoke(initial_state)
        st.session_state["result"] = result
        st.session_state["conversation_history"] = (
            result.get("conversation_history", [])
        )
        st.session_state["all_data_context"] = "\n\n".join(
            filter(None, [
                result.get("data_summary", ""),
                result.get("column_analysis", ""),
                result.get("trend_analysis", ""),
                result.get("forecast_analysis", ""),
                result.get("correlation_analysis", ""),
                result.get("anomalies", ""),
                result.get("comparison_analysis", "")
            ])
        )

        progress_bar.empty()
        status.empty()
        st.success("✅ Drelviq analysis complete!")

    # ── RESULTS ────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        chart_data = result.get("chart_data", {})

        # ── ROUTER DECISIONS ───────────────────────────────────────────────
        if result.get("router_reasoning"):
            st.markdown("<div class='custom-divider'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p class='section-header'>🧠 Smart Router Decisions</p>",
                unsafe_allow_html=True
            )
            lines = result["router_reasoning"].strip().split("\n")
            cols = st.columns(min(len(lines), 3))
            for i, line in enumerate(lines):
                with cols[i % 3]:
                    color = "#A5D6A7" if "✅" in line else "#EF9A9A"
                    st.markdown(
                        f"<div style='background:rgba(26,26,46,0.8);"
                        f"border:1px solid rgba(108,99,255,0.15);"
                        f"border-radius:8px;padding:10px 14px;"
                        f"font-size:0.78rem;color:{color};"
                        f"margin-bottom:8px;'>{line}</div>",
                        unsafe_allow_html=True
                    )

        # ── KPI CARDS ──────────────────────────────────────────────────────
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>📌 Key Metrics</p>",
            unsafe_allow_html=True
        )

        numeric_cols = result["numeric_cols"]
        kpi_items = [
            ("Total Rows", f"{len(df1):,}"),
            ("Columns", f"{len(df1.columns)}")
        ]
        for col in numeric_cols[:3]:
            val = df1[col].sum()
            kpi_items.append((
                col.replace("_", " ").title(),
                f"{val:,.0f}" if val >= 1 else f"{val:.4f}"
            ))

        kpi_cols = st.columns(len(kpi_items))
        for i, (label, value) in enumerate(kpi_items):
            with kpi_cols[i]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── COMPARISON CHARTS ──────────────────────────────────────────────
        comparison_charts = {
            k: v for k, v in chart_data.items()
            if k.startswith("comparison_")
        }

        if comparison_charts:
            st.markdown("<div class='custom-divider'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p class='section-header'>🔀 Dataset Comparison</p>",
                unsafe_allow_html=True
            )

            for chart_key, comp_data in comparison_charts.items():
                clean_title = (
                    chart_key.replace("comparison_", "")
                    .replace("_", " ").title()
                )
                categories = list(comp_data.keys())
                file1_vals = [comp_data[k]["file1"] for k in categories]
                file2_vals = [comp_data[k]["file2"] for k in categories]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="2024",
                    x=categories, y=file1_vals,
                    marker=dict(color=COLORS["primary"], opacity=0.85),
                    text=[f"{v:,.0f}" for v in file1_vals],
                    textposition="outside"
                ))
                fig.add_trace(go.Bar(
                    name="2025",
                    x=categories, y=file2_vals,
                    marker=dict(color=COLORS["secondary"], opacity=0.85),
                    text=[f"{v:,.0f}" for v in file2_vals],
                    textposition="outside"
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(
                        text=f"Comparison: {clean_title}",
                        font=dict(size=14, color="#E0E0E0")
                    ),
                    barmode="group",
                    height=380,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font=dict(color="#E0E0E0")
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            for chart_key, comp_data in list(comparison_charts.items())[:1]:
                categories = list(comp_data.keys())
                diffs = [
                    comp_data[k]["file2"] - comp_data[k]["file1"]
                    for k in categories
                ]
                colors_diff = [
                    COLORS["success"] if d > 0 else COLORS["danger"]
                    for d in diffs
                ]
                fig = go.Figure(go.Bar(
                    x=categories, y=diffs,
                    marker_color=colors_diff,
                    text=[
                        f"+{d:,.0f}" if d > 0 else f"{d:,.0f}"
                        for d in diffs
                    ],
                    textposition="outside"
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(
                        text="Δ Change: 2025 vs 2024",
                        font=dict(size=14, color="#E0E0E0")
                    ),
                    height=320
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── AUTO CHARTS ────────────────────────────────────────────────────
        regular_charts = {
            k: v for k, v in chart_data.items()
            if not k.startswith("forecast_")
            and not k.startswith("comparison_")
        }

        if regular_charts:
            st.markdown("<div class='custom-divider'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p class='section-header'>📊 Data Visualizations</p>",
                unsafe_allow_html=True
            )

            chart_items = list(regular_charts.items())
            chart_pairs = [
                chart_items[i:i+2]
                for i in range(0, len(chart_items), 2)
            ]

            for pair in chart_pairs:
                cols = st.columns(2)
                for idx, (chart_key, chart_values) in enumerate(pair):
                    with cols[idx]:
                        try:
                            chart_df = pd.DataFrame(
                                list(chart_values.items()),
                                columns=["Category", "Value"]
                            ).sort_values("Value", ascending=False)

                            title = chart_key.replace("_", " ").title()

                            if (
                                "monthly" in chart_key
                                or "quarterly" in chart_key
                                or "trend" in chart_key
                            ):
                                fig = px.line(
                                    chart_df,
                                    x="Category", y="Value",
                                    title=title, markers=True,
                                    color_discrete_sequence=[
                                        COLORS["primary"]
                                    ]
                                )
                                fig.update_traces(
                                    line=dict(width=2.5),
                                    marker=dict(size=6)
                                )

                            elif (
                                "channel" in chart_key
                                or "type" in chart_key
                                or "sector" in chart_key
                                or "stage" in chart_key
                                or "city" in chart_key
                            ) and len(chart_df) <= 6:
                                fig = px.pie(
                                    chart_df,
                                    values="Value",
                                    names="Category",
                                    title=title,
                                    color_discrete_sequence=COLORS["chart_seq"],
                                    hole=0.4
                                )
                                fig.update_traces(
                                    textposition="inside",
                                    textinfo="percent+label"
                                )

                            elif len(chart_df) <= 8:
                                chart_df = chart_df.sort_values(
                                    "Value", ascending=True
                                )
                                fig = px.bar(
                                    chart_df,
                                    x="Value", y="Category",
                                    orientation="h",
                                    title=title,
                                    color="Value",
                                    color_continuous_scale=[
                                        [0, "#1A1A2E"],
                                        [1, COLORS["primary"]]
                                    ],
                                    text_auto=".2s"
                                )
                                fig.update_layout(showlegend=False)

                            else:
                                chart_df = chart_df.sort_values(
                                    "Value", ascending=False
                                )
                                fig = px.bar(
                                    chart_df,
                                    x="Category", y="Value",
                                    title=title,
                                    color="Value",
                                    color_continuous_scale=[
                                        [0, "#1A1A2E"],
                                        [1, COLORS["secondary"]]
                                    ],
                                    text_auto=".2s"
                                )
                                fig.update_layout(showlegend=False)

                            fig.update_layout(
                                **PLOTLY_LAYOUT,
                                height=320,
                                title=dict(
                                    text=title,
                                    font=dict(size=13, color="#E0E0E0")
                                )
                            )
                            st.plotly_chart(
                                fig, use_container_width=True
                            )

                        except Exception as e:
                            st.warning(
                                f"Could not render {chart_key}: {e}"
                            )

        # ── FORECAST CHARTS ────────────────────────────────────────────────
        forecast_charts = {
            k: v for k, v in chart_data.items()
            if k.startswith("forecast_")
        }

        if forecast_charts:
            st.markdown("<div class='custom-divider'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p class='section-header'>🔮 Predictive Forecast</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size:0.8rem;color:#666;"
                "margin-top:-12px;margin-bottom:16px;'>"
                "Linear regression on historical data — "
                "blue is actual, orange is predicted</p>",
                unsafe_allow_html=True
            )

            for chart_key, chart_values in forecast_charts.items():
                col_name = chart_key.replace("forecast_", "")
                hist_labels = chart_values["historical_labels"]
                hist_values = chart_values["historical_values"]
                forecast_labels = chart_values["forecast_labels"]
                forecast_values = chart_values["forecast_values"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_labels, y=hist_values,
                    mode="lines+markers",
                    name="Historical",
                    line=dict(color=COLORS["primary"], width=2.5),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(108,99,255,0.08)"
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_labels, y=forecast_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(
                        color=COLORS["warning"],
                        width=2.5, dash="dash"
                    ),
                    marker=dict(
                        symbol="diamond", size=10,
                        color=COLORS["warning"]
                    )
                ))
                fig.add_vline(
                    x=hist_labels[-1],
                    line_dash="dot",
                    line_color="rgba(255,255,255,0.2)",
                    annotation_text="Forecast →",
                    annotation_font_color="#8B8FA8"
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(
                        text=f"{col_name.replace('_', ' ').title()} "
                        f"— Historical + 3 Month Forecast",
                        font=dict(size=14, color="#E0E0E0")
                    ),
                    height=420,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font=dict(color="#E0E0E0")
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── ANOMALY ALERTS ─────────────────────────────────────────────────
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>🚨 Anomaly Detection</p>",
            unsafe_allow_html=True
        )

        anomalies = result.get("anomalies", "")
        has_anomaly = "SPIKE" in anomalies or "DROP" in anomalies

        if has_anomaly:
            lines = anomalies.split("\n")
            for line in lines:
                if "SPIKE" in line:
                    st.markdown(
                        f"<div class='anomaly-spike'>"
                        f"{line.strip()}</div>",
                        unsafe_allow_html=True
                    )
                elif "DROP" in line:
                    st.markdown(
                        f"<div class='anomaly-drop'>"
                        f"{line.strip()}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                "<div class='anomaly-ok'>"
                "✅ No significant anomalies detected</div>",
                unsafe_allow_html=True
            )

        # ── AI REPORT ──────────────────────────────────────────────────────
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>🤖 Drelviq Intelligence Report</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='report-container'>"
            f"{result['final_report']}</div>",
            unsafe_allow_html=True
        )

        # ── CONVERSATIONAL MEMORY ──────────────────────────────────────────
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>💬 Ask Drelviq</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:0.8rem;color:#666;"
            "margin-top:-12px;margin-bottom:16px;'>"
            "Ask follow-up questions — Drelviq remembers "
            "full context</p>",
            unsafe_allow_html=True
        )

        history = st.session_state.get("conversation_history", [])
        if history:
            for turn in history:
                if turn["role"] == "user":
                    st.chat_message("user").write(turn["content"])
                elif turn["role"] == "assistant":
                    st.chat_message("assistant").write(turn["content"])

        follow_up = st.chat_input(
            "Ask Drelviq anything about your data..."
        )

        if follow_up and "all_data_context" in st.session_state:
            with st.spinner("Drelviq is analyzing..."):
                answer, updated_history = answer_with_memory(
                    follow_up,
                    st.session_state["all_data_context"],
                    st.session_state.get("conversation_history", [])
                )
                st.session_state["conversation_history"] = updated_history
            st.chat_message("user").write(follow_up)
            st.chat_message("assistant").write(answer)

        elif follow_up:
            st.warning(
                "⚠️ Run Drelviq analysis first "
                "before asking questions."
            )

        # ── DOWNLOAD + EMAIL ───────────────────────────────────────────────
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>📤 Export & Share</p>",
            unsafe_allow_html=True
        )

        history_text = ""
        if st.session_state.get("conversation_history"):
            history_text = (
                "\n\nCONVERSATION HISTORY\n" + "=" * 20 + "\n"
            )
            for turn in st.session_state["conversation_history"]:
                role = "You" if turn["role"] == "user" else "Drelviq"
                history_text += f"\n{role}: {turn['content']}\n"

        report_text = f"""
DRELVIQ INTELLIGENCE REPORT
============================

ROUTER DECISIONS
----------------
{result.get('router_reasoning', '')}

---
{result['final_report']}

---
FORECAST
--------
{result.get('forecast_analysis', '')}

---
COMPARISON
----------
{result.get('comparison_analysis', '')}

---
ANOMALIES
---------
{result.get('anomalies', '')}

---
DATA SUMMARY
------------
{result.get('data_summary', '')}
{history_text}
"""
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name="drelviq_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📊 Download Dataset",
                data=df1.to_csv(index=False),
                file_name="dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── EMAIL SECTION ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="email-card">
            <div style='font-size:0.9rem;font-weight:600;
                        color:#E0E0E0;margin-bottom:4px;'>
                📧 Send Report via Email
            </div>
            <div style='font-size:0.78rem;color:#666;margin-bottom:16px;'>
                Send the full Drelviq report to any email address
            </div>
        </div>
        """, unsafe_allow_html=True)

        email_col1, email_col2 = st.columns([3, 1])
        with email_col1:
            recipient_email = st.text_input(
                "Recipient email address",
                placeholder="boss@company.com",
                label_visibility="collapsed"
            )
        with email_col2:
            send_btn = st.button(
                "📧 Send Report",
                use_container_width=True,
                type="primary"
            )

        with st.expander("➕ Add more recipients", expanded=False):
            extra_emails = st.text_area(
                "Additional recipients (one per line)",
                placeholder="colleague@company.com\nteam@company.com",
                height=80,
                label_visibility="collapsed"
            )

        if send_btn:
            if not recipient_email:
                st.warning("⚠️ Please enter a recipient email address.")
            elif "@" not in recipient_email or "." not in recipient_email:
                st.error("❌ Please enter a valid email address.")
            else:
                recipients = [recipient_email.strip()]
                if extra_emails:
                    for email in extra_emails.strip().split("\n"):
                        email = email.strip()
                        if "@" in email and "." in email:
                            recipients.append(email)

                from emailer import send_report
                with st.spinner(
                    f"Sending report to {', '.join(recipients)}..."
                ):
                    success = send_report(
                        result, df1,
                        recipients=recipients,
                        subject="⚡ Drelviq Intelligence Report"
                    )

                if success:
                    st.success(
                        f"✅ Report sent to: {', '.join(recipients)}"
                    )
                else:
                    st.error(
                        "❌ Failed to send. Check RESEND_API_KEY in secrets."
                    )

        with st.expander("📄 View Full Dataset", expanded=False):
            st.dataframe(df1, use_container_width=True)

else:
    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:3rem;margin-bottom:16px;'>⚡</div>
        <div style='font-size:1.2rem;font-weight:600;
                    color:#E0E0E0;margin-bottom:8px;'>
            Welcome to Drelviq
        </div>
        <div style='font-size:0.875rem;color:#555;
                    max-width:400px;margin:0 auto;line-height:1.6;'>
            Upload any CSV or load the Indian Startup Ecosystem
            sample to see Drelviq in action
        </div>
    </div>
    """, unsafe_allow_html=True)
