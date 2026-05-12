import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("🚀 Loading user behavior data...")

df = pd.read_csv("data/processed/user_behavior.csv")

# -----------------------------
# CREATE RAW SCORE
# -----------------------------
df["raw_score"] = (
    df["total_spent"] * 0.4 +
    df["avg_spent"] * 0.3 +
    df["transaction_count"] * 0.3
)

# -----------------------------
# NORMALIZE TO 0-100
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 100))

df["financial_score"] = scaler.fit_transform(
    df[["raw_score"]]
)

# Round score
df["financial_score"] = df["financial_score"].round(2)

# -----------------------------
# CREATE CUSTOMER TIERS
# -----------------------------
def assign_tier(score):

    if score >= 80:
        return "Premium"

    elif score >= 60:
        return "Gold"

    elif score >= 40:
        return "Standard"

    else:
        return "Basic"

df["customer_tier"] = df["financial_score"].apply(assign_tier)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv(
    "data/processed/financial_scores.csv",
    index=False
)

print("✅ Financial scores generated!")

print(df[[
    "financial_score",
    "customer_tier"
]].head())