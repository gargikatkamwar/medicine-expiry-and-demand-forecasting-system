
import streamlit as st
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

def generate_risk_report():
    # Load data
    try:
        sales_df = pd.read_csv("medicine_sales_history.csv", parse_dates=["Date"])
        inv_df = pd.read_csv("medicine_inventory.csv", parse_dates=["ExpiryDate"])
    except FileNotFoundError:
        st.error("Error: 'medicine_sales_history.csv' or 'medicine_inventory.csv' not found. Please upload them.")
        return pd.DataFrame()

    st.sidebar.header("Data Overview")
    st.sidebar.write(f"Sales data shape: {sales_df.shape}")
    st.sidebar.write(f"Inventory data shape: {inv_df.shape}")

    # --- Prophet Forecasting ---
    st.sidebar.subheader("Forecasting Progress")
    forecast_results = {}
    medicine_names = sales_df["MedicineName"].unique()
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, med in enumerate(medicine_names):
        status_text.text(f"Forecasting for {med} ({i+1}/{len(medicine_names)})...")
        subset = sales_df[sales_df["MedicineName"] == med][["Date","UnitsSold"]]
        subset.columns = ["ds","y"]

        m = Prophet()
        m.fit(subset)

        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        # only keep future predictions (next 90 days)
        future_forecast = forecast[forecast["ds"] > subset["ds"].max()]
        predicted_demand = future_forecast["yhat"].clip(lower=0).sum()

        forecast_results[med] = predicted_demand
        progress_bar.progress((i + 1) / len(medicine_names))

    status_text.text("Forecasting complete!")

    forecast_df = pd.DataFrame(list(forecast_results.items()), columns=["MedicineName","Predicted90DayDemand"])

    # Merge with inventory
    inv_summary = inv_df.groupby("MedicineName").agg(
        TotalStock=("CurrentStock","sum"),
        NearestExpiry=("ExpiryDate","min"),
        UnitPrice=("UnitPrice","mean")
    ).reset_index()

    risk_df = inv_summary.merge(forecast_df, on="MedicineName")

    # Calculate DaysToExpiry
    today = sales_df["Date"].max()
    risk_df["DaysToExpiry"] = (risk_df["NearestExpiry"] - today).dt.days

    # Classify Risk
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

    return risk_df.sort_values("RiskLevel", ascending=False).reset_index(drop=True)

st.set_page_config(layout="wide")
st.title("Medicine Expiry Risk Report")

if st.button("Generate Report"):
    report_df = generate_risk_report()
    if not report_df.empty:
        st.subheader("Risk Analysis Results")
        st.dataframe(report_df)

        st.subheader("Risk Level Distribution")
        risk_counts = report_df["RiskLevel"].value_counts()
        st.bar_chart(risk_counts)

        # Optional: Display critical items
        st.subheader("Critical Items")
        critical_items = report_df[report_df["RiskLevel"] == "Critical"]
        if not critical_items.empty:
            st.dataframe(critical_items)
        else:
            st.write("No critical items identified.")

        # Download report
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Risk Report as CSV",
            data=csv,
            file_name="expiry_risk_report.csv",
            mime="text/csv",
        )

