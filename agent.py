# agent.py
# Autonomous Business Intelligence Agent
# LangGraph 10-node pipeline with smart routing
# Works locally (.env) and on Hugging Face (st.secrets)

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ── STATE ─────────────────────────────────────────────────────────────────────

class BIState(TypedDict):
    query: str
    df_json: str
    df2_json: str
    comparison_mode: bool
    columns: list
    numeric_cols: list
    categorical_cols: list
    date_cols: list
    run_trend: bool
    run_forecast: bool
    run_correlation: bool
    run_comparison: bool
    run_anomaly: bool
    router_reasoning: str
    data_summary: str
    column_analysis: str
    correlation_analysis: str
    trend_analysis: str
    forecast_analysis: str
    anomalies: str
    comparison_analysis: str
    insights: Annotated[list, operator.add]
    final_report: str
    chart_data: dict
    nl_answer: str
    conversation_history: list


# ── SMART COLUMN DETECTION ────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = []

    for col in categorical_cols:
        try:
            pd.to_datetime(df[col].head(10))
            date_cols.append(col)
        except:
            pass

    categorical_cols = [c for c in categorical_cols if c not in date_cols]

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols
    }


# ── SMART ROUTER ──────────────────────────────────────────────────────────────

def decide_analysis_plan(df: pd.DataFrame, col_types: dict,
                         comparison_mode: bool) -> dict:
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    date_cols = col_types["date"]
    n_rows = len(df)
    reasoning = []
    plan = {}

    if len(date_cols) > 0:
        plan["run_trend"] = True
        reasoning.append(
            f"✅ Trend analysis — date column '{date_cols[0]}' detected"
        )
    else:
        plan["run_trend"] = False
        reasoning.append(
            "❌ Trend analysis skipped — no date column detected"
        )

    if len(date_cols) > 0 and n_rows >= 50:
        plan["run_forecast"] = True
        reasoning.append(
            f"✅ Forecast — date column present, {n_rows} rows sufficient"
        )
    else:
        plan["run_forecast"] = False
        reasoning.append(
            f"❌ Forecast skipped — need date column and 50+ rows"
        )

    if len(numeric_cols) >= 2:
        plan["run_correlation"] = True
        reasoning.append(
            f"✅ Correlation — {len(numeric_cols)} numeric columns detected"
        )
    else:
        plan["run_correlation"] = False
        reasoning.append(
            f"❌ Correlation skipped — only {len(numeric_cols)} numeric column"
        )

    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        plan["run_anomaly"] = True
        reasoning.append(
            "✅ Anomaly detection — categorical and numeric columns present"
        )
    else:
        plan["run_anomaly"] = False
        reasoning.append(
            "❌ Anomaly detection skipped — need both categorical and numeric"
        )

    plan["run_comparison"] = comparison_mode
    if comparison_mode:
        reasoning.append("✅ Comparison — two datasets provided")
    else:
        reasoning.append("❌ Comparison skipped — single dataset mode")

    plan["router_reasoning"] = "\n".join(reasoning)
    return plan


# ── ANALYSIS TOOLS ────────────────────────────────────────────────────────────

def compute_data_summary(df: pd.DataFrame, numeric_cols: list,
                         categorical_cols: list) -> str:
    result = "DATASET OVERVIEW:\n"
    result += f"- Total rows: {len(df):,}\n"
    result += f"- Total columns: {len(df.columns)}\n"
    result += f"- Numeric columns: {numeric_cols}\n"
    result += f"- Categorical columns: {categorical_cols}\n\n"

    result += "NUMERIC COLUMN STATISTICS:\n"
    for col in numeric_cols:
        result += (
            f"- {col}: total={df[col].sum():,.2f} | "
            f"mean={df[col].mean():,.2f} | "
            f"min={df[col].min():,.2f} | "
            f"max={df[col].max():,.2f}\n"
        )

    result += "\nCATEGORICAL COLUMN VALUE COUNTS:\n"
    for col in categorical_cols[:5]:
        top_vals = df[col].value_counts().head(5)
        result += f"- {col}: {dict(top_vals)}\n"

    return result


def compute_column_analysis(df: pd.DataFrame, numeric_cols: list,
                            categorical_cols: list) -> tuple:
    result = "CATEGORICAL BREAKDOWN:\n"
    chart_data = {}

    for cat_col in categorical_cols[:3]:
        for num_col in numeric_cols[:2]:
            grouped = (
                df.groupby(cat_col)[num_col]
                .sum()
                .round(2)
                .sort_values(ascending=False)
            )
            result += f"\n{num_col} by {cat_col}:\n"
            for k, v in grouped.items():
                result += f"  - {k}: {v:,.2f}\n"
            chart_data[f"{num_col}_by_{cat_col}"] = grouped.to_dict()

    return result, chart_data


def compute_correlations(df: pd.DataFrame, numeric_cols: list) -> str:
    if len(numeric_cols) < 2:
        return "Not enough numeric columns for correlation analysis.\n"

    corr = df[numeric_cols].corr().round(3)
    result = "CORRELATION ANALYSIS:\n"
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                val = corr.loc[col1, col2]
                strength = (
                    "strong" if abs(val) > 0.7
                    else "moderate" if abs(val) > 0.4
                    else "weak"
                )
                direction = "positive" if val > 0 else "negative"
                result += (
                    f"  - {col1} vs {col2}: {val} "
                    f"({strength} {direction} correlation)\n"
                )
    return result


def compute_trends(df: pd.DataFrame, numeric_cols: list,
                   date_cols: list) -> tuple:
    result = "TREND ANALYSIS:\n"
    chart_data = {}

    if len(date_cols) > 0:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df["_month"] = df[date_col].dt.strftime("%Y-%m")
        df["_quarter"] = df[date_col].dt.to_period("Q").astype(str)

        for num_col in numeric_cols[:2]:
            monthly = df.groupby("_month")[num_col].sum().round(2)
            quarterly = df.groupby("_quarter")[num_col].sum().round(2)

            result += f"\nMonthly {num_col}:\n"
            for k, v in monthly.items():
                result += f"  - {k}: {v:,.2f}\n"

            result += f"\nQuarterly {num_col}:\n"
            for k, v in quarterly.items():
                result += f"  - {k}: {v:,.2f}\n"

            chart_data[f"monthly_{num_col}"] = monthly.to_dict()
            chart_data[f"quarterly_{num_col}"] = quarterly.to_dict()

    return result, chart_data


def compute_forecast(df: pd.DataFrame, numeric_cols: list,
                     date_cols: list) -> tuple:
    if len(date_cols) == 0 or len(numeric_cols) == 0:
        return "No date column found — skipping forecast.\n", {}

    from sklearn.linear_model import LinearRegression
    from pandas import Period

    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df["_month_num"] = df[date_col].dt.to_period("M").apply(
        lambda x: x.ordinal
    )

    result = "FORECAST (Next 3 Months):\n"
    chart_data = {}

    for num_col in numeric_cols[:2]:
        monthly = df.groupby("_month_num")[num_col].sum().reset_index()
        monthly.columns = ["month_num", "value"]

        if len(monthly) < 3:
            continue

        X = monthly["month_num"].values.reshape(-1, 1)
        y = monthly["value"].values

        model = LinearRegression()
        model.fit(X, y)

        last_month = monthly["month_num"].max()
        future_months = np.array([
            last_month + 1,
            last_month + 2,
            last_month + 3
        ]).reshape(-1, 1)

        predictions = model.predict(future_months)

        future_labels = [
            str(Period(ordinal=int(m), freq="M"))
            for m in future_months.flatten()
        ]

        result += f"\n{num_col} forecast:\n"
        for label, pred in zip(future_labels, predictions):
            result += f"  - {label}: {pred:,.2f} (predicted)\n"

        hist_labels = [
            str(Period(ordinal=int(m), freq="M"))
            for m in monthly["month_num"].values
        ]
        hist_values = monthly["value"].values.tolist()

        chart_data[f"forecast_{num_col}"] = {
            "historical_labels": hist_labels,
            "historical_values": hist_values,
            "forecast_labels": future_labels,
            "forecast_values": predictions.tolist()
        }

    return result, chart_data


def compute_anomalies(df: pd.DataFrame, numeric_cols: list,
                      categorical_cols: list) -> str:
    result = "ANOMALIES DETECTED:\n"

    for num_col in numeric_cols[:2]:
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            grouped = df.groupby(cat_col)[num_col].sum()
            mean_val = grouped.mean()
            std_val = grouped.std()

            for k, v in grouped.items():
                if v > mean_val + 1.5 * std_val:
                    result += (
                        f"  ⬆ SPIKE — {k}: {v:,.2f} "
                        f"(avg: {mean_val:,.2f}, "
                        f"+{((v-mean_val)/mean_val*100):.1f}%)\n"
                    )
                elif v < mean_val - 1.5 * std_val:
                    result += (
                        f"  ⬇ DROP — {k}: {v:,.2f} "
                        f"(avg: {mean_val:,.2f}, "
                        f"{((v-mean_val)/mean_val*100):.1f}%)\n"
                    )

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        result += "\nMISSING VALUES:\n"
        for col, count in missing.items():
            result += (
                f"  - {col}: {count} missing "
                f"({count/len(df)*100:.1f}%)\n"
            )
    else:
        result += "\nNo missing values detected.\n"

    return result


def compute_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                       numeric_cols: list,
                       categorical_cols: list) -> tuple:
    result = "DATASET COMPARISON (File 1 vs File 2):\n\n"
    chart_data = {}

    result += "OVERALL NUMERIC COMPARISON:\n"
    for col in numeric_cols:
        if col in df1.columns and col in df2.columns:
            val1 = df1[col].sum()
            val2 = df2[col].sum()
            diff = val2 - val1
            pct = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            arrow = "⬆" if diff > 0 else "⬇"
            result += (
                f"- {col}: "
                f"File1={val1:,.2f} | "
                f"File2={val2:,.2f} | "
                f"{arrow} {abs(pct):.1f}% "
                f"({'increase' if diff > 0 else 'decrease'})\n"
            )

    result += "\nCATEGORICAL BREAKDOWN COMPARISON:\n"
    for cat_col in categorical_cols[:2]:
        if cat_col not in df1.columns or cat_col not in df2.columns:
            continue
        for num_col in numeric_cols[:1]:
            if num_col not in df1.columns or num_col not in df2.columns:
                continue

            g1 = df1.groupby(cat_col)[num_col].sum().round(2)
            g2 = df2.groupby(cat_col)[num_col].sum().round(2)
            all_keys = set(g1.index) | set(g2.index)

            result += f"\n{num_col} by {cat_col}:\n"
            comp_data = {}
            for k in sorted(all_keys):
                v1 = g1.get(k, 0)
                v2 = g2.get(k, 0)
                diff = v2 - v1
                pct = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
                arrow = "⬆" if diff > 0 else "⬇"
                result += (
                    f"  - {k}: "
                    f"File1={v1:,.2f} → "
                    f"File2={v2:,.2f} "
                    f"({arrow} {abs(pct):.1f}%)\n"
                )
                comp_data[k] = {"file1": float(v1), "file2": float(v2)}

            chart_data[f"comparison_{num_col}_by_{cat_col}"] = comp_data

    result += f"\nROW COUNT:\n"
    result += f"- File 1: {len(df1):,} rows\n"
    result += f"- File 2: {len(df2):,} rows\n"
    result += f"- Difference: {len(df2) - len(df1):+,} rows\n"

    return result, chart_data


def answer_with_memory(query: str, data_summary: str,
                       conversation_history: list) -> tuple:
    if not query or query == "Generate a full business intelligence report":
        return "", conversation_history

    system_prompt = """You are a senior business intelligence analyst
with memory of the full conversation. Answer questions using ONLY
the data summary provided and previous conversation context.
Be specific, cite exact numbers, and reference previous answers
when relevant. Never make up data."""

    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(
        content=f"Here is the business data context:\n\n{data_summary}"
    ))
    messages.append(AIMessage(
        content="I have reviewed the data. Ready to answer questions."
    ))

    for turn in conversation_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    answer = response.content

    updated_history = conversation_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history


# ── AGENT NODES ───────────────────────────────────────────────────────────────

def detection_node(state: BIState) -> BIState:
    print("Detecting column types...")
    df = pd.read_json(state["df_json"])
    col_types = detect_columns(df)
    return {
        "numeric_cols": col_types["numeric"],
        "categorical_cols": col_types["categorical"],
        "date_cols": col_types["date"],
        "insights": []
    }


def router_node(state: BIState) -> BIState:
    print("Smart router deciding analysis plan...")
    df = pd.read_json(state["df_json"])
    col_types = {
        "numeric": state["numeric_cols"],
        "categorical": state["categorical_cols"],
        "date": state["date_cols"]
    }
    plan = decide_analysis_plan(
        df, col_types, state.get("comparison_mode", False)
    )
    print(f"\nRouter decisions:\n{plan['router_reasoning']}\n")
    return plan


def summary_node(state: BIState) -> BIState:
    print("Computing data summary...")
    df = pd.read_json(state["df_json"])
    summary = compute_data_summary(
        df, state["numeric_cols"], state["categorical_cols"]
    )
    return {"data_summary": summary, "insights": [summary]}


def column_analysis_node(state: BIState) -> BIState:
    print("Running column analysis...")
    df = pd.read_json(state["df_json"])
    analysis, chart_data = compute_column_analysis(
        df, state["numeric_cols"], state["categorical_cols"]
    )
    return {
        "column_analysis": analysis,
        "chart_data": chart_data,
        "insights": [analysis]
    }


def trend_node(state: BIState) -> BIState:
    if not state.get("run_trend"):
        print("Skipping trend analysis (router decision)")
        return {"trend_analysis": "", "insights": []}

    print("Running trend analysis...")
    df = pd.read_json(state["df_json"])
    trend, chart_data = compute_trends(
        df, state["numeric_cols"], state["date_cols"]
    )
    existing = state.get("chart_data", {})
    existing.update(chart_data)
    return {
        "trend_analysis": trend,
        "chart_data": existing,
        "insights": [trend]
    }


def correlation_node(state: BIState) -> BIState:
    if not state.get("run_correlation"):
        print("Skipping correlation analysis (router decision)")
        return {"correlation_analysis": "", "insights": []}

    print("Running correlation analysis...")
    df = pd.read_json(state["df_json"])
    corr = compute_correlations(df, state["numeric_cols"])
    return {
        "correlation_analysis": corr,
        "insights": [corr]
    }


def forecast_node(state: BIState) -> BIState:
    if not state.get("run_forecast"):
        print("Skipping forecast (router decision)")
        return {"forecast_analysis": "", "insights": []}

    print("Running forecast analysis...")
    df = pd.read_json(state["df_json"])
    forecast, chart_data = compute_forecast(
        df, state["numeric_cols"], state["date_cols"]
    )
    existing = state.get("chart_data", {})
    existing.update(chart_data)
    return {
        "forecast_analysis": forecast,
        "chart_data": existing,
        "insights": [forecast]
    }


def anomaly_node(state: BIState) -> BIState:
    if not state.get("run_anomaly"):
        print("Skipping anomaly detection (router decision)")
        return {"anomalies": "", "insights": []}

    print("Detecting anomalies...")
    df = pd.read_json(state["df_json"])
    anomalies = compute_anomalies(
        df, state["numeric_cols"], state["categorical_cols"]
    )
    return {"anomalies": anomalies, "insights": [anomalies]}


def comparison_node(state: BIState) -> BIState:
    if not state.get("run_comparison") or not state.get("df2_json"):
        print("Skipping comparison (router decision)")
        return {"comparison_analysis": "", "insights": []}

    print("Running comparison analysis...")
    df1 = pd.read_json(state["df_json"])
    df2 = pd.read_json(state["df2_json"])
    comparison, chart_data = compute_comparison(
        df1, df2,
        state["numeric_cols"],
        state["categorical_cols"]
    )
    existing = state.get("chart_data", {})
    existing.update(chart_data)
    return {
        "comparison_analysis": comparison,
        "chart_data": existing,
        "insights": [comparison]
    }


def report_node(state: BIState) -> BIState:
    print("Generating final AI report...")

    all_data = "\n\n".join(filter(None, [
        state.get("data_summary", ""),
        state.get("column_analysis", ""),
        state.get("trend_analysis", ""),
        state.get("forecast_analysis", ""),
        state.get("correlation_analysis", ""),
        state.get("anomalies", ""),
        state.get("comparison_analysis", "")
    ]))

    comparison_instruction = ""
    if state.get("comparison_mode"):
        comparison_instruction = (
            "7. COMPARISON SUMMARY "
            "(what changed between File 1 and File 2, "
            "what grew, what dropped, key differences)"
        )

    system_prompt = """You are a senior business intelligence analyst.
Every claim MUST be grounded in the data provided.
Never invent numbers. Cite exact figures.
Format with clear sections."""

    user_prompt = f"""
Based on this real business data, generate a comprehensive BI report.

{all_data}

Generate a report with:
1. EXECUTIVE SUMMARY (3-4 sentences with exact numbers)
2. KEY FINDINGS (5-7 bullet points with specific data citations)
3. ANOMALIES & ALERTS (unusual patterns and why they matter)
4. FORECAST INSIGHTS (what the predictions suggest)
5. RECOMMENDATIONS (3-5 actionable recommendations)
6. RISK AREAS (2-3 areas needing attention)
{comparison_instruction}

Be specific. Every claim must come from the data above.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    nl_answer = ""
    updated_history = state.get("conversation_history", [])

    if state.get("query") != "Generate a full business intelligence report":
        nl_answer, updated_history = answer_with_memory(
            state.get("query", ""),
            all_data,
            state.get("conversation_history", [])
        )

    return {
        "final_report": response.content,
        "nl_answer": nl_answer,
        "conversation_history": updated_history
    }


# ── BUILD GRAPH ───────────────────────────────────────────────────────────────

def build_bi_agent():
    graph = StateGraph(BIState)

    graph.add_node("detection", detection_node)
    graph.add_node("router", router_node)
    graph.add_node("summary", summary_node)
    graph.add_node("column_analysis", column_analysis_node)
    graph.add_node("trend", trend_node)
    graph.add_node("correlation", correlation_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("anomaly", anomaly_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("detection")
    graph.add_edge("detection", "router")
    graph.add_edge("router", "summary")
    graph.add_edge("summary", "column_analysis")
    graph.add_edge("column_analysis", "trend")
    graph.add_edge("trend", "correlation")
    graph.add_edge("correlation", "forecast")
    graph.add_edge("forecast", "anomaly")
    graph.add_edge("anomaly", "comparison")
    graph.add_edge("comparison", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "startups_data.csv"
    df = pd.read_csv(csv_path)

    agent = build_bi_agent()
    initial_state = {
        "query": "Generate a full business intelligence report",
        "df_json": df.to_json(),
        "df2_json": "",
        "comparison_mode": False,
        "columns": df.columns.tolist(),
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

    print("=" * 60)
    print("DRELVIQ — AUTONOMOUS BI AGENT")
    print("=" * 60)
    result = agent.invoke(initial_state)
    print(result["final_report"])
