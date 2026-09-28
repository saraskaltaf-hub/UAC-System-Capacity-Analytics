import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="UAC System Capacity Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("📊 System Capacity & Care Load Analytics")
st.subheader("Unaccompanied Children (UAC) Program")

st.write(
    "Interactive analytics dashboard for examining system load, "
    "CBP and HHS care levels, transfers, discharges and short-term "
    "system load forecasting."
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("UAC_Cleaned_Analytics_Dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

# =====================================================
# CALCULATE KPIs
# =====================================================

df["Total System Load"] = (
    df["Children in CBP custody"] +
    df["Children in HHS Care"]
)

df["Net Intake Pressure"] = (
    df["Children transferred out of CBP custody"] -
    df["Children discharged from HHS Care"]
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Dashboard Filters")

min_date = df["Date"].min()
max_date = df["Date"].max()

start_date = st.sidebar.date_input(
    "Start Date",
    min_date.date()
)

end_date = st.sidebar.date_input(
    "End Date",
    max_date.date()
)

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
].copy()

# =====================================================
# KPI CARDS
# =====================================================

st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average System Load",
        f"{filtered_df['Total System Load'].mean():,.0f}"
    )

with col2:
    st.metric(
        "Maximum System Load",
        f"{filtered_df['Total System Load'].max():,.0f}"
    )

with col3:
    st.metric(
        "Minimum System Load",
        f"{filtered_df['Total System Load'].min():,.0f}"
    )

with col4:
    st.metric(
        "Avg. Net Intake Pressure",
        f"{filtered_df['Net Intake Pressure'].mean():,.2f}"
    )

st.divider()

# =====================================================
# CHART 1: TOTAL SYSTEM LOAD
# =====================================================

st.header("1. Total System Care Load")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    filtered_df["Date"],
    filtered_df["Total System Load"],
    label="Total System Load"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("Total UAC System Care Load Over Time")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# =====================================================
# CHART 2: CBP VS HHS
# =====================================================

st.header("2. CBP Custody vs HHS Care")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    filtered_df["Date"],
    filtered_df["Children in CBP custody"],
    label="CBP Custody"
)

ax.plot(
    filtered_df["Date"],
    filtered_df["Children in HHS Care"],
    label="HHS Care"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("CBP Custody vs HHS Care")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# =====================================================
# CHART 3: TRANSFERS VS DISCHARGES
# =====================================================

st.header("3. Transfers vs Discharges")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    filtered_df["Date"],
    filtered_df["Children transferred out of CBP custody"],
    label="Transfers to HHS"
)

ax.plot(
    filtered_df["Date"],
    filtered_df["Children discharged from HHS Care"],
    label="HHS Discharges"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("HHS Transfers vs Discharges")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# =====================================================
# CHART 4: NET INTAKE PRESSURE
# =====================================================

st.header("4. Net Intake Pressure")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    filtered_df["Date"],
    filtered_df["Net Intake Pressure"],
    label="Net Intake Pressure"
)

ax.axhline(0, linestyle="--")

ax.set_xlabel("Date")
ax.set_ylabel("Children")
ax.set_title("Net Intake Pressure Over Time")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# =====================================================
# FORECAST
# =====================================================

st.header("5. Short-Term System Load Forecast")

forecast_df = df[
    ["Date", "Total System Load"]
].dropna().copy()

forecast_df["Days"] = (
    forecast_df["Date"] -
    forecast_df["Date"].min()
).dt.days

X = forecast_df[["Days"]]
y = forecast_df["Total System Load"]

forecast_model = LinearRegression()
forecast_model.fit(X, y)

last_day = forecast_df["Days"].max()

future_days = np.arange(
    last_day + 1,
    last_day + 31
)

future_X = pd.DataFrame({
    "Days": future_days
})

future_predictions = forecast_model.predict(
    future_X
)

forecast_dates = pd.date_range(
    start=forecast_df["Date"].max() +
    pd.Timedelta(days=1),
    periods=30
)

forecast_result = pd.DataFrame({
    "Date": forecast_dates,
    "Forecasted System Load": future_predictions
})

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast_df["Date"],
    forecast_df["Total System Load"],
    label="Historical Load"
)

ax.plot(
    forecast_result["Date"],
    forecast_result["Forecasted System Load"],
    linestyle="--",
    label="30-Day Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Children")
ax.set_title("UAC System Load Forecast")
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# =====================================================
# FORECAST TABLE
# =====================================================

st.subheader("Forecasted System Load")

st.dataframe(
    forecast_result,
    use_container_width=True
)

# =====================================================
# DATA TABLE
# =====================================================

st.header("Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# FOOTER


st.divider()

st.caption(
    "UAC System Capacity & Care Load Analytics | "
    "Predictive Analytics Project"
)