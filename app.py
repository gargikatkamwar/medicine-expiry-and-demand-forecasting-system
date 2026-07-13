import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Medicine Expiry & Demand Dashboard", layout="wide")
st.title("💊 Medicine Expiry & Demand Forecasting Dashboard")

@st.cache_data
def load_data():
    sales_df = pd.read_csv("medicine_sales_history.csv", parse_dates=["Date"])
    inv_df = pd.read_csv("medicine_inventory.csv", parse_dates=["ExpiryDate"])
    return sales_df, inv_df

sales_df, inv_df = load_data()

@st.cache_data
def compute_risk_report(sales_df, inv_df):
    forecast_results = {}
    for med in sales_df["MedicineName"].unique():
        subset = sales_df[sales_df["MedicineName"] == med][["Date","UnitsSold"]]
        subset.columns = ["ds","y"]
        m = Prophet()
        m.fit(subset)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        future_forecast = forecast[forecast["ds"] > subset["ds"].max()]
        forecast_results[med] = future_forecast["yhat"].clip(lower=0).sum()

    forecast_df = pd.DataFrame(list(forecast_results.items()), columns=["MedicineName","Predicted90DayDemand"])

    inv_summary = inv_df.groupby("MedicineName").agg(
        TotalStock=("CurrentStock","sum"),
        NearestExpiry=("ExpiryDate","min"),
        UnitPrice=("UnitPrice","mean")
    ).reset_index()

    risk_df = inv_summary.merge(forecast_df, on="MedicineName")
    today = sales_df["Date"].max()
    risk_df["DaysToExpiry"] = (risk_df["NearestExpiry"] - today).dt.days

    def classify_risk(row):
        if row["DaysToExpiry"] > 90:
            return "Safe"
        if row["Predicted90DayDemand"] >= row["TotalStock"]:
            return "Safe"
        elif row["Predicted90DayDemand"] >= row["TotalStock"] * 0.6:
            return "At Risk"
        else:
            return "Critical"

    risk_df["RiskLevel"] = risk_df.apply(classify_risk, axis=1)
    risk_df["SurplusUnits"] = (risk_df["TotalStock"] - risk_df["Predicted90DayDemand"]).clip(lower=0)
    risk_df["PotentialLoss"] = risk_df["SurplusUnits"] * risk_df["UnitPrice"]
    return risk_df

with st.spinner("Running forecasts across all medicines..."):
    risk_df = compute_risk_report(sales_df, inv_df)

# --- Summary cards ---
col1, col2, col3 = st.columns(3)
col1.metric("Safe", (risk_df["RiskLevel"]=="Safe").sum())
col2.metric("At Risk", (risk_df["RiskLevel"]=="At Risk").sum())
col3.metric("Critical", (risk_df["RiskLevel"]=="Critical").sum())

st.subheader("Risk Report")
risk_filter = st.multiselect("Filter by Risk Level", options=risk_df["RiskLevel"].unique(), default=risk_df["RiskLevel"].unique())
st.dataframe(risk_df[risk_df["RiskLevel"].isin(risk_filter)].sort_values("PotentialLoss", ascending=False))

st.subheader("Total Potential Loss from Expiring Stock")
st.write(f"₹{risk_df['PotentialLoss'].sum():,.2f}")

# --- Individual medicine forecast chart ---
st.subheader("Demand Forecast for a Specific Medicine")
selected_med = st.selectbox("Select Medicine", sales_df["MedicineName"].unique())

subset = sales_df[sales_df["MedicineName"] == selected_med][["Date","UnitsSold"]]
subset.columns = ["ds","y"]
m = Prophet()
m.fit(subset)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

fig = m.plot(forecast)
st.pyplot(fig)
