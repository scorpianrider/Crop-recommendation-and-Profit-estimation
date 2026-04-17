# training/train_rainfall_model.py
#
# FIX: Target column is "future_rainfall_mm" (was "future_rainfall_cm" which
#      caused a KeyError because preprocessing outputs "future_rainfall_mm").

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocessing.rainfall_preprocessing import preprocess_rainfall_data


def train_rainfall_model():
    print("Preprocessing rainfall data...")
    df = preprocess_rainfall_data()

    feature_columns = [
        "district_code", "current_rainfall", "current_month", "harvest_month",
        "normal_current_rainfall", "normal_harvest_rainfall", "rainfall_anomaly"
    ]

    X = df[feature_columns]
    y = df["future_rainfall_mm"]   # ← FIXED: was "future_rainfall_cm"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=14,
        min_samples_split=4, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE : {round(mean_absolute_error(y_test, y_pred), 4)} mm")
    print(f"R²  : {round(r2_score(y_test, y_pred), 4)}")
    print(f"Predicted range: {y_pred.min():.1f} – {y_pred.max():.1f} mm")
    print(f"Actual range   : {y_test.min():.1f} – {y_test.max():.1f} mm")

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "rainfall_api_model.pkl"))
    print("Rainfall model saved.")


if __name__ == "__main__":
    train_rainfall_model()