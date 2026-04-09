# emailer.py
# Drelviq — sends HTML BI report via Resend API
# Works locally (.env) and on Hugging Face (st.secrets)

import os
import resend
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

resend.api_key = os.getenv(
    "RESEND_API_KEY") or st.secrets.get("RESEND_API_KEY", "")
EMAIL_RECEIVER = os.getenv(
    "EMAIL_RECEIVER") or st.secrets.get("EMAIL_RECEIVER", "")


def generate_html_report(result: dict, df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    numeric_cols = result.get("numeric_cols", [])
    anomalies = result.get("anomalies", "")
    final_report = result.get("final_report", "")
    forecast = result.get("forecast_analysis", "")
    router = result.get("router_reasoning", "")

    kpi_html = ""
    kpi_items = [
        ("Total Rows", f"{len(df):,}"),
        ("Columns", f"{len(df.columns)}")
    ]
    for col in numeric_cols[:3]:
        val = df[col].sum()
        kpi_items.append((
            col.replace("_", " ").title(),
            f"{val:,.0f}"
        ))

    for label, value in kpi_items:
        kpi_html += f"""
        <div style="background:#1E1E35;
                    border:1px solid rgba(108,99,255,0.3);
                    border-radius:10px;padding:16px 20px;
                    text-align:center;display:inline-block;
                    margin:6px;min-width:120px;">
            <div style="font-size:1.5rem;font-weight:700;
                        color:#6C63FF;">{value}</div>
            <div style="font-size:0.75rem;color:#888;
                        margin-top:4px;text-transform:uppercase;
                        letter-spacing:0.5px;">{label}</div>
        </div>"""

    anomaly_html = ""
    has_anomaly = "SPIKE" in anomalies or "DROP" in anomalies
    if has_anomaly:
        for line in anomalies.split("\n"):
            if "SPIKE" in line:
                anomaly_html += f"""
                <div style="background:rgba(244,67,54,0.1);
                            border-left:3px solid #F44336;
                            border-radius:0 8px 8px 0;
                            padding:10px 16px;margin:6px 0;
                            font-size:0.875rem;color:#EF9A9A;">
                    {line.strip()}</div>"""
            elif "DROP" in line:
                anomaly_html += f"""
                <div style="background:rgba(255,152,0,0.1);
                            border-left:3px solid #FF9800;
                            border-radius:0 8px 8px 0;
                            padding:10px 16px;margin:6px 0;
                            font-size:0.875rem;color:#FFCC80;">
                    {line.strip()}</div>"""
    else:
        anomaly_html = """
        <div style="background:rgba(76,175,80,0.1);
                    border-left:3px solid #4CAF50;
                    border-radius:0 8px 8px 0;
                    padding:10px 16px;
                    font-size:0.875rem;color:#A5D6A7;">
            ✅ No significant anomalies detected</div>"""

    report_html = ""
    for line in final_report.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**"):
            report_html += f"""
            <h3 style="color:#6C63FF;font-size:1rem;
                       margin:20px 0 8px 0;">
                {line.replace('**', '')}</h3>"""
        elif line.startswith("*") or line.startswith("-"):
            report_html += f"""
            <div style="padding:4px 0 4px 16px;
                        border-left:2px solid rgba(108,99,255,0.3);
                        margin:4px 0;color:#C0C0C0;
                        font-size:0.875rem;">
                {line.lstrip('*- ')}</div>"""
        else:
            report_html += f"""
            <p style="color:#C0C0C0;font-size:0.875rem;
                      line-height:1.6;margin:8px 0;">
                {line}</p>"""

    forecast_html = ""
    if forecast and "skipping" not in forecast.lower():
        for line in forecast.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "forecast" in line.lower() and ":" in line:
                forecast_html += f"""
                <h4 style="color:#FF9800;font-size:0.9rem;
                           margin:12px 0 6px 0;">{line}</h4>"""
            elif line.startswith("-"):
                forecast_html += f"""
                <div style="padding:3px 0 3px 16px;
                            color:#C0C0C0;font-size:0.85rem;">
                    {line}</div>"""

    router_html = ""
    for line in router.split("\n"):
        if not line.strip():
            continue
        color = "#A5D6A7" if "✅" in line else "#EF9A9A"
        router_html += f"""
        <div style="font-size:0.78rem;color:{color};
                    padding:3px 0;">{line.strip()}</div>"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#0F0F1A;
             font-family:-apple-system,BlinkMacSystemFont,
             'Segoe UI',sans-serif;">
<div style="max-width:700px;margin:0 auto;padding:24px;">

    <div style="background:linear-gradient(135deg,#1A1A2E,#16213E,#0F3460);
                border:1px solid rgba(108,99,255,0.3);
                border-radius:16px;padding:32px;margin-bottom:24px;">
        <div style="font-size:2rem;font-weight:800;color:#fff;
                    letter-spacing:-1px;">⚡ Drelviq</div>
        <div style="font-size:0.75rem;color:#6C63FF;letter-spacing:2px;
                    text-transform:uppercase;margin-top:2px;">
            Intelligence Report</div>
        <p style="color:#8B8FA8;margin:12px 0 0 0;font-size:0.9rem;">
            Generated on {now}</p>
        <div style="margin-top:16px;">
            <span style="background:rgba(108,99,255,0.15);
                         border:1px solid rgba(108,99,255,0.4);
                         color:#A09BFF;padding:4px 12px;
                         border-radius:20px;font-size:0.75rem;
                         margin-right:8px;">⚡ LangGraph Pipeline</span>
            <span style="background:rgba(108,99,255,0.15);
                         border:1px solid rgba(108,99,255,0.4);
                         color:#A09BFF;padding:4px 12px;
                         border-radius:20px;font-size:0.75rem;
                         margin-right:8px;">🚫 Zero Hallucination</span>
            <span style="background:rgba(108,99,255,0.15);
                         border:1px solid rgba(108,99,255,0.4);
                         color:#A09BFF;padding:4px 12px;
                         border-radius:20px;font-size:0.75rem;">
                🔮 AI Forecasted</span>
        </div>
    </div>

    <div style="background:#1A1A2E;
                border:1px solid rgba(108,99,255,0.15);
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <h2 style="color:#E0E0E0;font-size:1rem;
                   font-weight:600;margin:0 0 16px 0;">
            📌 Key Metrics</h2>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
            {kpi_html}</div>
    </div>

    <div style="background:#1A1A2E;
                border:1px solid rgba(108,99,255,0.15);
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <h2 style="color:#E0E0E0;font-size:1rem;
                   font-weight:600;margin:0 0 16px 0;">
            🚨 Anomaly Alerts</h2>
        {anomaly_html}
    </div>

    {"" if not forecast_html else f'''
    <div style="background:#1A1A2E;
                border:1px solid rgba(108,99,255,0.15);
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <h2 style="color:#E0E0E0;font-size:1rem;
                   font-weight:600;margin:0 0 16px 0;">
            🔮 Forecast — Next 3 Months</h2>
        {forecast_html}
    </div>'''}

    <div style="background:#1A1A2E;
                border:1px solid rgba(108,99,255,0.15);
                border-radius:12px;padding:24px;margin-bottom:20px;">
        <h2 style="color:#E0E0E0;font-size:1rem;
                   font-weight:600;margin:0 0 16px 0;">
            🤖 Drelviq Intelligence Report</h2>
        {report_html}
    </div>

    {"" if not router_html else f'''
    <div style="background:#1A1A2E;
                border:1px solid rgba(108,99,255,0.15);
                border-radius:12px;padding:20px;margin-bottom:20px;">
        <h2 style="color:#E0E0E0;font-size:1rem;
                   font-weight:600;margin:0 0 16px 0;">
            🧠 Smart Router Decisions</h2>
        {router_html}
    </div>'''}

    <div style="text-align:center;padding:20px;
                color:#444;font-size:0.75rem;">
        <p style="margin:0;">
            Generated by ⚡ Drelviq · Powered by LangGraph + Groq
        </p>
        <p style="margin:4px 0 0 0;">
            Every insight grounded in real data — zero hallucination
        </p>
    </div>

</div>
</body>
</html>
"""
    return html


def send_report(result: dict, df: pd.DataFrame,
                recipients: list = None,
                subject: str = None) -> bool:
    if not resend.api_key:
        print("RESEND_API_KEY not set")
        return False

    if not recipients:
        recipients = [EMAIL_RECEIVER]

    recipients = [
        r for r in recipients
        if r and "@" in r and "." in r
    ]

    if not recipients:
        print("No valid recipients provided")
        return False

    now = datetime.now().strftime("%B %d, %Y")
    subject = subject or f"⚡ Drelviq Intelligence Report — {now}"
    html_content = generate_html_report(result, df)

    try:
        response = resend.Emails.send({
            "from": "Drelviq BI <onboarding@resend.dev>",
            "to": recipients,
            "subject": subject,
            "html": html_content
        })
        print(f"Report sent to {recipients} — ID: {response['id']}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
