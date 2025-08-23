import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------
# Load Data
# ------------------------
pred_df = pd.read_csv("deliverables/predictions_3m.csv")
acc_df = pd.read_csv("reports/final_accuracy.csv")

st.title("🚗 Dealership KPI Forecasting Dashboard")

# ------------------------
# KPI Forecast Viewer
# ------------------------
st.header("KPI Forecasts")
kpi = st.selectbox("Select KPI", pred_df["english_name"].unique())

df_kpi = pred_df[pred_df["english_name"] == kpi]
fig = px.line(df_kpi, x="ds", y="y_hat", title=f"3-Month Forecast for {kpi}")
st.plotly_chart(fig)

# ------------------------
# Accuracy Summary
# ------------------------
st.header("Accuracy Analysis")

st.subheader("MAPE Distribution")
fig2 = px.histogram(acc_df, x="MAPE", nbins=30, title="MAPE Histogram", marginal="box")
st.plotly_chart(fig2)

st.subheader("Top 5 Best KPIs")
best_kpis = acc_df.sort_values("MAPE").head(5)
st.dataframe(best_kpis[["english_name","MAPE"]])

st.subheader("Top 5 Worst KPIs")
worst_kpis = acc_df.sort_values("MAPE", ascending=False).head(5)
st.dataframe(worst_kpis[["english_name","MAPE"]])

# ------------------------
# Correlation Heatmap
# ------------------------
st.header("Correlation Heatmap")
import seaborn as sns
import matplotlib.pyplot as plt

# Load feature matrix and show correlation heatmap for top 20 columns
feature_df = pd.read_csv("features/feature_matrix.csv")
# Select only numeric columns (exclude dates, strings, etc.)
numeric_cols = feature_df.select_dtypes(include=['number']).columns[:20]
if len(numeric_cols) == 0:
    st.warning("No numeric columns found in feature matrix for correlation heatmap.")
else:
    corr = feature_df[numeric_cols].corr()
    fig3, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# ------------------------
# Scenario Engine (What-if)
# ------------------------
st.header("What-if Scenario")
# User selects model (KPI) and month for scenario
model_options = pred_df["english_name"].unique()
selected_model = st.selectbox("Select Model (KPI) for Scenario", model_options, key="scenario_model")
month_options = pred_df["ds"].unique()
selected_month = st.selectbox("Select Month for Scenario", month_options, key="scenario_month")
change = st.slider("Change Sales (%)", -20, 20, 0)

# Only run scenario if change is not zero
if change != 0:
    st.write(f"If {selected_model} in {selected_month} changes by {change}%:")
    # Get correlation matrix from feature_df (numeric columns)
    corr_matrix = feature_df.select_dtypes(include=['number']).corr()
    # Find correlations with the selected KPI
    if selected_model in corr_matrix.columns:
        related_kpis = corr_matrix[selected_model].drop(selected_model).sort_values(ascending=False)
        # Show top 5 most correlated KPIs
        st.subheader("Estimated effect on other KPIs (top 5 by correlation):")
        effect_data = []
        for kpi, corr_val in related_kpis.head(5).items():
            # Estimate % change in other KPI as: corr * change
            est_change = corr_val * change
            effect_data.append({"KPI": kpi, "Correlation": corr_val, "Estimated % Change": est_change})
        st.dataframe(pd.DataFrame(effect_data))
        # Update predictions for the selected month and next 2 months
        st.subheader("Updated predictions for next 3 months:")
        # Get rows for selected KPI and next 3 months
        pred_rows = pred_df[(pred_df["english_name"] == selected_model)].copy()
        # Find index of selected month
        month_idx = pred_rows[pred_rows["ds"] == selected_month].index.min()
        if pd.isna(month_idx):
            st.warning("Selected month not found in predictions.")
        else:
            # Update y_hat for selected month and next 2 months
            for i in range(month_idx, month_idx+3):
                if i in pred_rows.index:
                    pred_rows.at[i, "y_hat"] *= (1 + change/100)
            st.dataframe(pred_rows.loc[month_idx:month_idx+2, ["ds", "y_hat"]])
            fig4 = px.line(pred_rows.loc[month_idx-2:month_idx+5], x="ds", y="y_hat", title=f"Scenario: {selected_model} with {change}% change")
            st.plotly_chart(fig4)
    else:
        st.warning(f"No correlation data for {selected_model}.")
   
