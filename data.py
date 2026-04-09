# data.py
# Veltrix — Indian Startup Ecosystem Dataset
# Generates two realistic datasets for comparison
# Dataset 1 — 2024 (baseline) | Dataset 2 — 2025 (growth)
# Covers funding, revenue, hiring, burn rate across sectors and cities

import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]
sectors = ["Fintech", "Edtech", "Healthtech", "SaaS", "D2C", "Deeptech"]
stages = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]
channels = ["VC Funded", "Bootstrapped", "Angel Funded"]

base_funding = {
    "Fintech": 5000000,
    "Edtech": 2000000,
    "Healthtech": 3000000,
    "SaaS": 4000000,
    "D2C": 1500000,
    "Deeptech": 6000000
}

base_revenue = {
    "Fintech": 800000,
    "Edtech": 400000,
    "Healthtech": 600000,
    "SaaS": 1200000,
    "D2C": 350000,
    "Deeptech": 200000
}


def generate_dataset(year: int, scenario: str) -> pd.DataFrame:
    rows = []
    start_date = datetime(year, 1, 1)

    for i in range(500):
        date = start_date + timedelta(days=random.randint(0, 364))
        city = random.choice(cities)
        sector = random.choice(sectors)
        stage = random.choice(stages)
        channel = random.choice(channels)

        funding = base_funding[sector] * random.uniform(0.5, 2.5)
        revenue = base_revenue[sector] * random.uniform(0.3, 2.0)
        employees = random.randint(5, 500)
        burn_rate = funding * random.uniform(0.05, 0.25)

        if scenario == "baseline":
            # 2024 patterns
            # Edtech struggling post-covid
            if sector == "Edtech":
                funding *= 0.6
                revenue *= 0.5

            # Bengaluru dominates
            if city == "Bengaluru":
                funding *= 1.3
                revenue *= 1.2

            # SaaS Q4 spike
            if sector == "SaaS" and date.month in [10, 11, 12]:
                revenue *= 1.5

            # Deeptech underfunded
            if sector == "Deeptech" and stage in ["Pre-Seed", "Seed"]:
                funding *= 0.4

        elif scenario == "growth":
            # 2025 patterns — AI wave hits India
            # Deeptech and SaaS surge
            if sector in ["Deeptech", "SaaS"]:
                funding *= 1.8
                revenue *= 1.6
                employees = int(employees * 1.4)

            # Fintech consolidation
            if sector == "Fintech":
                funding *= 0.85
                revenue *= 1.3

            # Edtech recovery
            if sector == "Edtech":
                funding *= 1.2
                revenue *= 1.4

            # Hyderabad rising — T-Hub effect
            if city == "Hyderabad":
                funding *= 1.4
                revenue *= 1.3

            # Bengaluru still dominant
            if city == "Bengaluru":
                funding *= 1.5
                revenue *= 1.4

            # Series B and C mega rounds
            if stage in ["Series B", "Series C"]:
                funding *= 2.2

            # D2C brand boom
            if sector == "D2C" and date.month in [10, 11, 12]:
                revenue *= 2.0
                employees = int(employees * 1.3)

            # Bootstrapped startups growing faster
            if channel == "Bootstrapped":
                revenue *= 1.4

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "city": city,
            "sector": sector,
            "stage": stage,
            "channel": channel,
            "funding_usd": round(funding, 2),
            "revenue_usd": round(revenue, 2),
            "employees": int(employees),
            "burn_rate_usd": round(burn_rate, 2)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── GENERATE BOTH DATASETS ────────────────────────────────────────────────────

print("Generating Dataset 1 — 2024 (Baseline)...")
df_2024 = generate_dataset(2024, "baseline")
df_2024.to_csv("startups_2024.csv", index=False)

print("Generating Dataset 2 — 2025 (Growth)...")
df_2025 = generate_dataset(2025, "growth")
df_2025.to_csv("startups_2025.csv", index=False)

df_2025.to_csv("startups_data.csv", index=False)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VELTRIX — INDIAN STARTUP ECOSYSTEM DATA")
print("=" * 60)

print(f"\n2024 (Baseline):")
print(f"  Rows: {len(df_2024):,}")
print(f"  Total Funding: ${df_2024['funding_usd'].sum():,.0f}")
print(f"  Total Revenue: ${df_2024['revenue_usd'].sum():,.0f}")
print(f"  Total Employees: {df_2024['employees'].sum():,}")

print(f"\n2025 (Growth):")
print(f"  Rows: {len(df_2025):,}")
print(f"  Total Funding: ${df_2025['funding_usd'].sum():,.0f}")
print(f"  Total Revenue: ${df_2025['revenue_usd'].sum():,.0f}")
print(f"  Total Employees: {df_2025['employees'].sum():,}")

rev_diff = df_2025['revenue_usd'].sum() - df_2024['revenue_usd'].sum()
rev_pct = (rev_diff / df_2024['revenue_usd'].sum()) * 100
print(f"\nYEAR OVER YEAR:")
print(f"  Revenue change: ${rev_diff:+,.0f} ({rev_pct:+.1f}%)")

print(f"\n2024 by Sector (Revenue):")
s2024 = df_2024.groupby(
    "sector")["revenue_usd"].sum().sort_values(ascending=False)
for s, v in s2024.items():
    print(f"  {s}: ${v:,.0f}")

print(f"\n2025 by Sector (Revenue):")
s2025 = df_2025.groupby(
    "sector")["revenue_usd"].sum().sort_values(ascending=False)
for s, v in s2025.items():
    print(f"  {s}: ${v:,.0f}")

print(f"\n2024 by City (Funding):")
c2024 = df_2024.groupby(
    "city")["funding_usd"].sum().sort_values(ascending=False)
for c, v in c2024.items():
    print(f"  {c}: ${v:,.0f}")

print(f"\n2025 by City (Funding):")
c2025 = df_2025.groupby(
    "city")["funding_usd"].sum().sort_values(ascending=False)
for c, v in c2025.items():
    print(f"  {c}: ${v:,.0f}")

print("\n" + "=" * 60)
print("Files generated:")
print("  startups_2024.csv — use as File 1")
print("  startups_2025.csv — use as File 2")
print("  startups_data.csv — default single dataset")
print("=" * 60)
