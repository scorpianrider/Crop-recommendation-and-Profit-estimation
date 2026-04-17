# training/train_fertilizer_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_fertilizer_model():
    print("Loading Fertilizer Dataset...")

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    model_dir    = os.path.join(current_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(datasets_dir, "Fertilizer_data.csv"))

    required = ["Crop", "Soil_Colour", "Soil_Texture",
                "Nitrogen", "Phosphorous", "Potassium", "Fertilizer"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df[required].dropna()

    crop_encoder    = LabelEncoder()
    colour_encoder  = LabelEncoder()
    texture_encoder = LabelEncoder()
    label_encoder   = LabelEncoder()

    df["Crop_Enc"]    = crop_encoder.fit_transform(df["Crop"])
    df["Colour_Enc"]  = colour_encoder.fit_transform(df["Soil_Colour"])
    df["Texture_Enc"] = texture_encoder.fit_transform(df["Soil_Texture"])
    df["Label_Enc"]   = label_encoder.fit_transform(df["Fertilizer"])

    X = df[["Crop_Enc", "Colour_Enc", "Texture_Enc",
            "Nitrogen", "Phosphorous", "Potassium"]]
    y = df["Label_Enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "RandomForest":       RandomForestClassifier(random_state=42),
    }

    results = {}
    print("\nTraining Fertilizer Models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"  Accuracy: {acc:.4f}")

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    best_model.fit(X, y)

    joblib.dump(best_model,      os.path.join(model_dir, "fertilizer_model.pkl"))
    joblib.dump(crop_encoder,    os.path.join(model_dir, "fertilizer_crop_encoder.pkl"))
    joblib.dump(colour_encoder,  os.path.join(model_dir, "fertilizer_colour_encoder.pkl"))
    joblib.dump(texture_encoder, os.path.join(model_dir, "fertilizer_texture_encoder.pkl"))
    joblib.dump(label_encoder,   os.path.join(model_dir, "fertilizer_label_encoder.pkl"))
    joblib.dump(results,         os.path.join(model_dir, "fertilizer_results.pkl"))
    print(f"\nBest Fertilizer Model: {best_name} — saved.")


if __name__ == "__main__":
    train_fertilizer_model()
