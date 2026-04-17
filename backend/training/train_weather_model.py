# training/train_weather_model.py

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocessing.weather_preprocessing import preprocess_weather_data


def train_weather_models():
    print("Loading and preprocessing weather dataset...")
    df = preprocess_weather_data()

    if len(df) > 120000:
        df = df.sample(n=120000, random_state=42)
        print(f"Sampled to {df.shape} for speed")

    feature_columns = [
        "district_code", "current_temp", "current_humidity",
        "current_month", "harvest_month",
        "normal_current_temp", "normal_current_humidity",
        "normal_harvest_temp", "normal_harvest_humidity",
        "temp_anomaly", "humidity_anomaly"
    ]

    X          = df[feature_columns]
    y_temp     = df["future_temp"]
    y_humidity = df["future_humidity"]

    X_train, X_test, y_temp_train, y_temp_test, y_hum_train, y_hum_test = train_test_split(
        X, y_temp, y_humidity, test_size=0.2, random_state=42
    )

    params = dict(n_estimators=300, max_depth=14, min_samples_split=4,
                  min_samples_leaf=2, random_state=42, n_jobs=-1)

    temp_model     = RandomForestRegressor(**params)
    humidity_model = RandomForestRegressor(**params)

    print("Training temperature model...")
    temp_model.fit(X_train, y_temp_train)
    print("Training humidity model...")
    humidity_model.fit(X_train, y_hum_train)

    print("\n=== Temperature Model ===")
    print("MAE:", round(mean_absolute_error(y_temp_test, temp_model.predict(X_test)), 4))
    print("R2 :", round(r2_score(y_temp_test, temp_model.predict(X_test)), 4))

    print("\n=== Humidity Model ===")
    print("MAE:", round(mean_absolute_error(y_hum_test, humidity_model.predict(X_test)), 4))
    print("R2 :", round(r2_score(y_hum_test, humidity_model.predict(X_test)), 4))

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(temp_model,     os.path.join(model_dir, "weather_temperature_model.pkl"))
    joblib.dump(humidity_model, os.path.join(model_dir, "weather_humidity_model.pkl"))
    print("\nWeather models saved.")


if __name__ == "__main__":
    train_weather_models()
