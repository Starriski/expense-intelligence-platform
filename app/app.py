import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Expense Intelligence Platform",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_transactions.csv")
    return df

df = load_data()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("deployment/fraud_model.pkl")
    return model

model = load_model()

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("💳 Expense Intelligence Platform")

st.markdown("""
AI-powered fintech analytics dashboard for:
- Fraud Detection
- Customer Segmentation
- Financial Intelligence Scoring
""")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

if "category" in df.columns:

    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df["category"].unique(),
        default=df["category"].unique()
    )

    df = df[df["category"].isin(categories)]

# -----------------------------
# KPI METRICS
# -----------------------------
st.subheader("📊 Key Metrics")

total_transactions = len(df)

total_amount = df["amount"].sum()

fraud_count = df["is_fraud"].sum()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Total Spend",
    f"${total_amount:,.2f}"
)

col3.metric(
    "Fraud Transactions",
    f"{fraud_count:,}"
)

# -----------------------------
# TRANSACTION PREVIEW
# -----------------------------
st.subheader("🧾 Transaction Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# -----------------------------
# SPENDING BY CATEGORY
# -----------------------------
st.subheader("💰 Spending by Category")

if "category" in df.columns:

    category_data = (
        df.groupby("category")["amount"]
        .sum()
        .reset_index()
        .sort_values(by="amount", ascending=False)
    )

    fig = px.bar(
        category_data,
        x="category",
        y="amount",
        title="Total Spending by Category",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------
# FRAUD DISTRIBUTION
# -----------------------------
st.subheader("🚨 Fraud Distribution")

fraud_counts = (
    df["is_fraud"]
    .value_counts()
    .reset_index()
)

fraud_counts.columns = ["Fraud Status", "Count"]

fraud_counts["Fraud Status"] = fraud_counts[
    "Fraud Status"
].map({
    0: "Normal",
    1: "Fraud"
})

fig = px.pie(
    fraud_counts,
    names="Fraud Status",
    values="Count",
    title="Fraud vs Normal Transactions"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# FRAUD PREDICTION TOOL
# -----------------------------
st.subheader("🤖 Fraud Prediction Tool")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

city_pop = st.number_input(
    "City Population",
    min_value=0,
    value=50000
)

unix_time = st.number_input(
    "Unix Time",
    min_value=0,
    value=1371816865
)

merchant_lat = st.number_input(
    "Merchant Latitude",
    value=40.0
)

merchant_long = st.number_input(
    "Merchant Longitude",
    value=-73.0
)

input_data = pd.DataFrame({
    "amount": [amount],
    "city_pop": [city_pop],
    "unix_time": [unix_time],
    "merch_lat": [merchant_lat],
    "merch_long": [merchant_long]
})

if st.button("Predict Fraud Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    if prediction == 1:

        st.error(
            f"⚠ Fraudulent Transaction Detected "
            f"(Risk Score: {probability:.2f})"
        )

    else:

        st.success(
            f"✅ Normal Transaction "
            f"(Risk Score: {probability:.2f})"
        )

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------
st.subheader("👥 Customer Segmentation")

segments_df = pd.read_csv(
    "deployment/customer_segments.csv"
)

segment_chart = (
    segments_df["segment"]
    .value_counts()
    .reset_index()
)

segment_chart.columns = ["Segment", "Users"]

fig = px.bar(
    segment_chart,
    x="Segment",
    y="Users",
    title="Customer Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    segments_df.head(50),
    use_container_width=True
)

# -----------------------------
# FINANCIAL HEALTH SCORES
# -----------------------------
st.subheader("💰 Financial Intelligence Scores")

score_df = pd.read_csv(
    "deployment/financial_scores.csv"
)

tier_chart = (
    score_df["customer_tier"]
    .value_counts()
    .reset_index()
)

tier_chart.columns = ["Tier", "Users"]

fig = px.funnel(
    tier_chart,
    x="Users",
    y="Tier",
    title="Customer Tier Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

top_customers = score_df.sort_values(
    by="financial_score",
    ascending=False
).head(20)

st.dataframe(
    top_customers,
    use_container_width=True
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    "Built with ❤️ using Streamlit, Machine Learning, and Plotly"
)