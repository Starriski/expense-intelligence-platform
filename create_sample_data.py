import pandas as pd

# Load full dataset
df = pd.read_csv("data/processed/transactions_final.csv")

# Take small sample
sample_df = df.sample(5000, random_state=42)

# Save sample
sample_df.to_csv(
    "data/sample_transactions.csv",
    index=False
)

print("Sample dataset created!")