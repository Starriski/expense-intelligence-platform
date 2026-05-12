import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    # Time features
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.day_name()

    # Spending intensity bucket
    df["spend_type"] = df["amount"].apply(
        lambda x: "high" if x > 1000 else ("medium" if x > 300 else "low")
    )

    # Weekend flag
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)

    print("[FEATURE ENGINEERING DONE]")

    return df