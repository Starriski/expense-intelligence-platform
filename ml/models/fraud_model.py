import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier


def train_fraud_model():

    print("\n Loading dataset...")

    df = pd.read_csv("data/processed/transactions_final.csv")

    print("Original shape:", df.shape)

    #  STRATIFIED SAMPLING
  
    df, _ = train_test_split(
        df,
        train_size=200000,
        random_state=42,
        stratify=df["is_fraud"]
    )

    print("Sampled shape:", df.shape)

  
    # FEATURES
    
    features = [
        "amount",
        "city_pop",
        "lat",
        "long",
        "merch_lat",
        "merch_long"
    ]

    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["is_fraud"]

    # TRAIN TEST SPLIT
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # IMPROVED MODEL (BOOSTING)
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    #PROBABILITY-BASED PREDICTION
    
    y_probs = model.predict_proba(X_test)[:, 1]

    # THRESHOLD TUNING 
    threshold = 0.3
    y_pred = (y_probs > threshold).astype(int)

    #EVALUATION
    
    print("\nFRAUD DETECTION RESULTS:\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:", roc_auc_score(y_test, y_probs))

    return model


if __name__ == "__main__":
    train_fraud_model()