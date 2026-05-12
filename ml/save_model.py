import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


print("🚀 Loading dataset...")

df = pd.read_csv("data/processed/transactions_final.csv")

# Smaller training sample

df, _ = train_test_split(
    df,
    train_size=50000,
    random_state=42,
    stratify=df["is_fraud"]
)

features = [
    "amount",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long"
]

X = df[features]
y = df["is_fraud"]

print("🧠 Training model...")

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3
)

model.fit(X, y)

# Save model
joblib.dump(model, "ml/fraud_model.pkl")

print("✅ Model saved successfully!")