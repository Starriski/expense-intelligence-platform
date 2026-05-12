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
# TITLE
# -----------------------------
st.title("💳 Expense Intelligence Platform")
st.markdown("### AI-Powered Fraud Detection + Financial Analytics Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_transactions.csv")
    return df

df = load_data()

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("ml/fraud_model.pkl")
    return model

model = load_model()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["category"].dropna().unique())
)

if category_filter:
    df = df[df["category"].isin(category_filter)]

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{len(df):,}"
)

col2.metric(
    "Fraud Cases",
    f"{int(df['is_fraud'].sum()):,}"
)

fraud_rate = round(df["is_fraud"].mean() * 100, 2)

col3.metric(
    "Fraud Rate %",
    fraud_rate
)

col4.metric(
    "Total Amount",
    f"${df['amount'].sum():,.2f}"
)

# -----------------------------
# TRANSACTION DATA
# -----------------------------
st.subheader("📄 Transaction Data Preview")

preview_columns = [
    col for col in [
        "date",
        "category",
        "merchant",
        "amount",
        "city",
        "state",
        "is_fraud"
    ]
    if col in df.columns
]

st.dataframe(
    df[preview_columns].head(100),
    use_container_width=True
)

# -----------------------------
# CATEGORY SPENDING
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
st.subheader("🚨 Fraud vs Normal Transactions")

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
st.subheader("🧠 Live Fraud Prediction")

st.markdown("Enter transaction details below to estimate fraud probability.")

col1, col2, col3 = st.columns(3)

amount = col1.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

city_pop = col2.number_input(
    "City Population",
    min_value=0,
    value=10000
)

lat = col3.number_input(
    "Customer Latitude",
    value=40.0
)

long = col1.number_input(
    "Customer Longitude",
    value=-75.0
)

merch_lat = col2.number_input(
    "Merchant Latitude",
    value=40.5
)

merch_long = col3.number_input(
    "Merchant Longitude",
    value=-75.5
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Fraud Risk"):

    input_data = np.array([
        [
            amount,
            city_pop,
            lat,
            long,
            merch_lat,
            merch_long
        ]
    ])

    fraud_probability = model.predict_proba(input_data)[0][1]

    st.markdown(f"## 🔴 Fraud Probability: {fraud_probability:.2f}")

    if fraud_probability > 0.7:
        st.error("⚠️ HIGH FRAUD RISK")

    elif fraud_probability > 0.3:
        st.warning("⚠️ MODERATE FRAUD RISK")

    else:
        st.success("✅ LOW FRAUD RISK")

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------
st.subheader("👥 Customer Segmentation")

segments_df = pd.read_csv(
    "data/processed/customer_segments.csv"
)

segment_counts = (
    segments_df["segment"]
    .value_counts()
    .sort_index()
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
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "Built using Python, Streamlit, Scikit-learn, Pandas, and ML-based Fraud Analytics"
)