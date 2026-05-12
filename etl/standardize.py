import numpy as np
import pandas as pd

def standardize_columns(df):

    df = df.copy()

    df = df.rename(columns={
        "amt": "amount",
        "trans_date_trans_time": "date",
        "trans_num": "transaction_id"
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["user_id"] = np.random.randint(1, 5000, size=len(df))

    print("[STANDARDIZATION DONE]")
    print("Columns:", df.columns)

    return df