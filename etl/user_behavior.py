import pandas as pd
import numpy as np

def build_user_behavior(df):

    # 1. Aggregate user-level stats
    user_stats = df.groupby("user_id").agg({
        "amount": ["sum", "mean", "count"]
    })

    user_stats.columns = ["total_spent", "avg_spent", "transaction_count"]
    user_stats = user_stats.reset_index()

    # 2. Spending intensity score
    user_stats["spending_intensity"] = (
        user_stats["total_spent"] / (user_stats["transaction_count"] + 1)
    )

    # 3. Risk segmentation (simple rule-based baseline)
    user_stats["risk_segment"] = pd.cut(
        user_stats["spending_intensity"],
        bins=[0, 500, 1500, 5000, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    )

    print("[USER BEHAVIOR ENGINE COMPLETED]")

    return user_stats