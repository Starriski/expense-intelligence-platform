#import pandas as pd

##from feature_engineering import add_features
#from etl.user_behavior import build_user_behavior

import pandas as pd

from etl.cleaning import clean_transactions
from etl.feature_engineering import add_features
from etl.synthetic_expander import expand_to_real_world
from etl.user_behavior import build_user_behavior
from etl.standardize import standardize_columns


def run_pipeline(file_path):
    print("\n[PIPELINE START]\n")

    df = pd.read_csv(file_path)
    print("Raw shape:", df.shape)

    # 1. STANDARDIZE FIRST (CRITICAL FIX)
    df = standardize_columns(df)

    # 2. Clean
    df = clean_transactions(df)

    # 3. Feature engineering
    df = add_features(df)

    # 4. User behavior
    user_df = build_user_behavior(df)

    # 5. Save outputs
    df.to_csv("data/processed/transactions_final.csv", index=False)
    user_df.to_csv("data/processed/user_behavior.csv", index=False)

    print("\n[PIPELINE COMPLETE]")


if __name__ == "__main__":
    run_pipeline("data/raw/credit_card.csv")