import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def expand_to_real_world(df, n_users=50):

    print("[EXPANSION STARTED]")

    # Assign user IDs
    df["user_id"] = np.random.randint(1, n_users, size=len(df))

    # Generate realistic dates
    start_date = datetime(2025, 1, 1)

    df["date"] = [
        start_date + timedelta(days=int(i % 365))
        for i in range(len(df))
    ]

    # Merchant simulation
    merchants = [
        "Amazon", "Flipkart", "Swiggy", "Zomato",
        "Uber", "Ola", "Netflix", "Spotify",
        "Electricity Board", "Rent Payment", "ATM Withdrawal"
    ]

    df["merchant"] = [random.choice(merchants) for _ in range(len(df))]

    # Category mapping
    category_map = {
        "Amazon": "Shopping",
        "Flipkart": "Shopping",
        "Swiggy": "Food",
        "Zomato": "Food",
        "Uber": "Transport",
        "Ola": "Transport",
        "Netflix": "Subscription",
        "Spotify": "Subscription",
        "Electricity Board": "Utilities",
        "Rent Payment": "Rent",
        "ATM Withdrawal": "Cash"
    }

    df["category"] = df["merchant"].map(category_map)

    # Payment modes
    df["payment_mode"] = np.random.choice(
        ["UPI", "Credit Card", "Debit Card"],
        size=len(df)
    )

    print("[EXPANSION COMPLETED] Shape:", df.shape)

    return df