# training/train_irrigation_model.py

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Allow importing data module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.generate_irrigation_dataset import generate_dataset


def train_irrigation_model():
    print("Generating Irrigation Dataset...")
    df = generate_dataset()

    crop_encoder     = LabelEncoder()
    wr_encoder       = LabelEncoder()
    drainage_encoder = LabelEncoder()
    label_encoder    = LabelEncoder()

    df["Crop_Enc"]     = crop_encoder.fit_transform(df["Crop"])
    df["WR_Enc"]       = wr_encoder.fit_transform(df["Water_Retention"])
    df["Drainage_Enc"] = drainage_encoder.fit_transform(df["Drainage"])
    df["Label_Enc"]    = label_encoder.fit_transform(df["Irrigation_Type"])

    X = df[["Crop_Enc", "WR_Enc", "Drainage_Enc"]]
    y = df["Label_Enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "RandomForest":       RandomForestClassifier(random_state=42),
    }

    results = {}
    print("\nTraining Irrigation Models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"  Accuracy: {acc:.4f}")

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    best_model.fit(X, y)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model,       os.path.join(model_dir, "irrigation_model.pkl"))
    joblib.dump(crop_encoder,     os.path.join(model_dir, "irrigation_crop_encoder.pkl"))
    joblib.dump(wr_encoder,       os.path.join(model_dir, "irrigation_wr_encoder.pkl"))
    joblib.dump(drainage_encoder, os.path.join(model_dir, "irrigation_drainage_encoder.pkl"))
    joblib.dump(label_encoder,    os.path.join(model_dir, "irrigation_label_encoder.pkl"))
    joblib.dump(results,          os.path.join(model_dir, "irrigation_results.pkl"))
    print(f"\nBest Irrigation Model: {best_name} — saved.")


if __name__ == "__main__":
    train_irrigation_model()
