# training/train_market_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def train_market_model():
    print("Loading Market Price Dataset...")

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    model_dir    = os.path.join(current_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(datasets_dir, "combined_price_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"combined_price_data.csv not found at {csv_path}\n"
            "Run:  python -m data.merge_market_data   first."
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["harvest_month"] = df["date"].dt.month
    df["harvest_year"]  = df["date"].dt.year

    encoder = LabelEncoder()
    df["crop_encoded"] = encoder.fit_transform(df["crop"])

    X = df[["crop_encoded", "harvest_month", "harvest_year"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Market Price Model R²: {score:.4f}")

    joblib.dump(model,   os.path.join(model_dir, "price_model.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "crop_encoder.pkl"))
    print("Market price model saved.")


if __name__ == "__main__":
    train_market_model()
