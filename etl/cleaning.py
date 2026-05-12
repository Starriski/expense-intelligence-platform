import pandas as pd
import numpy as np

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"] = df["amount"].fillna(df["amount"].median())

    if "merchant" in df.columns:
        df["merchant"] = df["merchant"].fillna("unknown")

    # Date parsing (safe handling)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # Remove invalid values
    if "amount" in df.columns:
        df = df[df["amount"] > 0]

    df = df.reset_index(drop=True)

    print("[CLEANING DONE] Shape:", df.shape)

    return df
