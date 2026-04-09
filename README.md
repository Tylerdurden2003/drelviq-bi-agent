# ⚡ Drelviq

Drelviq is an autonomous business intelligence agent. You upload a CSV — it figures out what to analyze, runs the analysis, and gives you a report. No configuration, no SQL, no dashboarding skills required.

Built with LangGraph, Groq, and Streamlit.

---

## What it does

Most BI tools make you ask questions. Drelviq asks them for you.

When you upload a dataset, a 10-node LangGraph pipeline runs autonomously:

- Inspects your columns and decides which analyses actually make sense for your data
- Breaks down your numbers by every category it finds
- Detects monthly and quarterly trends if a date column exists
- Flags statistical anomalies — spikes and drops that fall outside normal range
- Forecasts the next 3 months using linear regression on historical data
- If you upload two files, it compares them and shows exactly what changed
- Synthesizes everything into a grounded AI report where every claim is cited from real data

You can also ask follow-up questions in plain English. The agent remembers the full conversation and answers from the actual data — not from its training weights.

When you're done, send the full report to any email address directly from the dashboard.

---

## Why zero hallucination matters

Most LLM-powered analytics tools let the model reason freely about data. That means it can invent trends, misremember numbers, or confidently state things that aren't in the dataset.

Drelviq works differently. All analysis runs in pandas first. The LLM only receives computed numbers as context — it never touches the raw data. Every claim in the report traces back to a specific pandas calculation. If something isn't in the data, the agent says so.

---

## Tech stack

- **LangGraph** — 10-node sequential pipeline with smart routing
- **Groq API** — llama-3.1-8b-instant at temperature 0 (deterministic)
- **Pandas + NumPy** — all data analysis runs here, not in the LLM
- **scikit-learn** — LinearRegression for forecasting
- **Plotly** — interactive charts, auto-selected based on data type
- **Streamlit** — dashboard UI
- **Resend** — email delivery

---

## Setup

```bash
git clone https://github.com/Tylerdurden2003/drelviq-bi-agent
cd drelviq-bi-agent
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
RESEND_API_KEY=your_resend_key
EMAIL_RECEIVER=your@email.com
```

Generate the sample dataset:

```bash
python data.py
```

Run:

```bash
streamlit run app.py
```

---

## Sample data

The built-in sample covers the Indian startup ecosystem — 500 data points across Bengaluru, Mumbai, Delhi, Hyderabad, Pune, and Chennai. Six sectors: Fintech, Edtech, Healthtech, SaaS, D2C, and Deeptech. Two versions available (2024 and 2025) for year-over-year comparison.

You can replace this with any CSV — sales data, HR records, financial reports, operations data. Drelviq auto-detects the column types and adapts the analysis accordingly.

---

## Project structure

```
drelviq-bi-agent/
├── agent.py          # LangGraph 10-node pipeline
├── app.py            # Streamlit dashboard
├── data.py           # Sample dataset generator
├── emailer.py        # Resend email integration
├── startups_data.csv # Default sample dataset
├── startups_2024.csv # Baseline for comparison
├── startups_2025.csv # Comparison dataset
└── requirements.txt
```

---

## Decisions worth explaining

**Smart routing** — the pipeline doesn't blindly run all 10 nodes. It first checks what columns exist and decides what's relevant. No date column means trend analysis and forecasting are skipped. Only one numeric column means correlation is skipped. This keeps the output clean and relevant regardless of what you upload.

**temperature=0** — the LLM runs at zero temperature. Same dataset, same report every time. Consistency matters more than creativity in analytics.

**Chart selection** — time series data gets line charts. Small category breakdowns get donut charts. Ranked comparisons get horizontal bars. Larger datasets get vertical bars. The agent picks based on the data shape, not randomly.
