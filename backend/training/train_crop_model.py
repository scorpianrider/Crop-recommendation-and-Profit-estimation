# training/train_crop_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_crop_model():
    print("Loading Crop Recommendation Dataset...")

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    model_dir    = os.path.join(current_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(datasets_dir, "Crop recommendation data.csv"))
    df.columns = df.columns.str.strip().str.lower()

    required = ["temperature", "humidity", "rainfall", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required].dropna()
    X  = df[["temperature", "humidity", "rainfall"]]
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "RandomForest":       RandomForestClassifier(random_state=42),
    }

    results = {}
    print("\nTraining Crop Models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"  Accuracy: {acc:.4f}")

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    best_model.fit(X, y)  # retrain on full data

    joblib.dump(best_model, os.path.join(model_dir, "crop_model.pkl"))
    joblib.dump(results,    os.path.join(model_dir, "crop_results.pkl"))
    print(f"\nBest Crop Model: {best_name} — saved.")


if __name__ == "__main__":
    train_crop_model()
