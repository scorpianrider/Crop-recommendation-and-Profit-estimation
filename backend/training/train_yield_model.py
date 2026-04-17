# training/train_yield_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

DISTRICT_ALIASES = {
    "Thiruvarur":    "Tiruvarur",
    "Sivaganga":     "Sivagangai",
    "The Nilgiris":  "Nilgiris",
    "Villupuram":    "Viluppuram",
    "Kanniyakumari": "Kanyakumari",
}
CROP_ALIASES = {
    "beet root":    "beetroot",
    "water melon":  "watermelon",
    "pump kin":     "pumpkin",
    "sweet potato": "sweet potato",
}


def normalize_district(name):
    name = name.strip().title()
    return DISTRICT_ALIASES.get(name, name)


def normalize_crop(name):
    name = name.strip().lower()
    return CROP_ALIASES.get(name, name)


def train_yield_model():
    print("Loading Yield Dataset...")

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    model_dir    = os.path.join(current_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(datasets_dir, "Yield_data.csv"))
    df.columns = df.columns.str.strip()

    for col in ["District_Name", "Crop", "Area", "Production"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[["District_Name", "Crop", "Area", "Production"]].dropna()
    df = df[df["Area"] > 0]

    df["District_Name"] = df["District_Name"].apply(normalize_district)
    df["Crop"]          = df["Crop"].apply(normalize_crop)
    df["Yield"]         = df["Production"] / df["Area"]
    df = df[(df["Yield"] > 0) & (df["Yield"] < 100)]

    X = pd.get_dummies(df[["District_Name", "Crop"]])
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree":     DecisionTreeRegressor(random_state=42),
        "RandomForest":     RandomForestRegressor(random_state=42),
    }

    results = {}
    print("\nTraining Yield Models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2    = r2_score(y_test, preds)
        results[name] = {"RMSE": mean_squared_error(y_test, preds)**0.5, "R2": r2}
        print(f"  R2={r2:.4f}")

    best_name  = max(results, key=lambda x: results[x]["R2"])
    best_model = models[best_name]
    best_model.fit(X, y)

    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "yield_feature_columns.pkl"))
    joblib.dump(best_model,         os.path.join(model_dir, "yield_model.pkl"))
    joblib.dump(results,            os.path.join(model_dir, "yield_results.pkl"))
    print(f"\nBest Yield Model: {best_name} — saved.")


if __name__ == "__main__":
    train_yield_model()
