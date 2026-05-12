import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 Loading user behavior data...")

df = pd.read_csv("data/processed/user_behavior.csv")

# -----------------------------
# FEATURES
# -----------------------------
features = [
    "total_spent",
    "avg_spent",
    "transaction_count"
]

X = df[features]

# SCALE DATA

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# KMEANS CLUSTERING

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["segment"] = kmeans.fit_predict(X_scaled)


# SAVE OUTPUT

df.to_csv(
    "data/processed/customer_segments.csv",
    index=False
)

# Save model + scaler
joblib.dump(kmeans, "ml/kmeans_model.pkl")
joblib.dump(scaler, "ml/scaler.pkl")

print("Customer segmentation completed!")
print(df["segment"].value_counts())