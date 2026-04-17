# preprocessing/weather_preprocessing.py
#
# KEY FIXES:
#   1. TMP_2m is in Kelvin (e.g. 301.69 K = 28.5°C) → ALWAYS convert to
#      Celsius by subtracting 273.15. The old mean()>200 check was unreliable.
#   2. RH_2m is already 0–100 % — just clamp, don't rescale.
#   3. Cross-month anomaly features are built AFTER Kelvin→Celsius conversion
#      so they are correct (was broken before: OpenWeather gives °C but
#      climate normals were stored in Kelvin → anomaly was off by ~273).

import pandas as pd
import os
import joblib
from config.allowed_districts import ALLOWED_DISTRICTS


def preprocess_weather_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.join(current_dir, "..")
    data_path   = os.path.join(base_dir, "..", "datasets", "Weather_dataset.csv")
    models_dir  = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    print("Columns in Weather_dataset.csv:", df.columns.tolist())

    if "District" not in df.columns:
        raise ValueError("Weather_dataset.csv must have a 'District' column")

    df["District"] = df["District"].astype(str).str.strip()
    df = df[df["District"].isin(ALLOWED_DISTRICTS)].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after filtering ALLOWED_DISTRICTS. "
            "Check district names in ALLOWED_DISTRICTS match the CSV."
        )

    print(f"Filtered weather data shape: {df.shape}")

    # ── Column aliases ────────────────────────────────────────────────────────
    rename_map = {}
    if "TMP_2m" in df.columns and "Temperature" not in df.columns:
        rename_map["TMP_2m"] = "Temperature"
    if "RH_2m" in df.columns and "Humidity" not in df.columns:
        rename_map["RH_2m"] = "Humidity"
    df.rename(columns=rename_map, inplace=True)

    # ── Month column ──────────────────────────────────────────────────────────
    if "month" not in df.columns:
        if "time" in df.columns:
            df["time"]  = pd.to_datetime(df["time"], errors="coerce")
            df["month"] = df["time"].dt.month
        else:
            raise ValueError(
                "Weather_dataset.csv must have a 'month' or 'time' column"
            )

    for col in ["District", "Temperature", "Humidity", "month"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing after rename.")

    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Humidity"]    = pd.to_numeric(df["Humidity"],    errors="coerce")
    df["month"]       = pd.to_numeric(df["month"],       errors="coerce")

    # ── FIX 1: Kelvin → Celsius (unconditional if values > 200) ──────────────
    # Values like 301.69 are Kelvin. We check max(), not mean(), because
    # a small dataset could have mean < 200 even in Kelvin.
    if df["Temperature"].max() > 200:
        print(
            f"  → Converting Temperature Kelvin→Celsius "
            f"(max={df['Temperature'].max():.1f} K)"
        )
        df["Temperature"] = df["Temperature"] - 273.15
    else:
        print(f"  → Temperature already in Celsius (max={df['Temperature'].max():.1f})")

    # ── FIX 2: Humidity clamp ─────────────────────────────────────────────────
    # RH_2m is 0–100 in this dataset. Just guard against stray bad rows.
    df["Humidity"] = df["Humidity"].clip(0.0, 100.0)

    df.dropna(subset=["District", "Temperature", "Humidity", "month"], inplace=True)

    # Sanity filter after Kelvin conversion
    df = df[
        (df["month"] >= 1)         & (df["month"] <= 12)   &
        (df["Humidity"] >= 0)      & (df["Humidity"] <= 100) &
        (df["Temperature"] >= -10) & (df["Temperature"] <= 60)
    ].copy()

    print(f"Rows after sanity filter: {len(df)}")

    # ── District encoder ──────────────────────────────────────────────────────
    unique_districts = sorted(df["District"].unique().tolist())
    district_encoder = {d: i for i, d in enumerate(unique_districts)}
    joblib.dump(
        district_encoder,
        os.path.join(models_dir, "weather_district_encoder.pkl")
    )
    print(f"Saved weather_district_encoder with {len(district_encoder)} districts")

    df["district_code"] = df["District"].map(district_encoder)

    # ── Monthly averages per district (in °C and %) ───────────────────────────
    monthly_avg = (
        df.groupby(["District", "district_code", "month"], as_index=False)[
            ["Temperature", "Humidity"]
        ]
        .mean()
        .rename(columns={"Temperature": "avg_temp", "Humidity": "avg_humidity"})
    )

    # ── Climate lookup ────────────────────────────────────────────────────────
    climate_lookup: dict = {}
    for _, row in monthly_avg.iterrows():
        d = row["District"]
        m = int(row["month"])
        climate_lookup.setdefault(d, {})[m] = {
            "temp":     float(row["avg_temp"]),     # °C
            "humidity": float(row["avg_humidity"]), # %
        }

    joblib.dump(
        climate_lookup,
        os.path.join(models_dir, "weather_monthly_climate.pkl")
    )
    print("Saved weather_monthly_climate.pkl")

    # Quick sanity print so you can verify values look right
    print("\n── Climate sanity check (should be °C 15–45, humidity 30–100) ──")
    for d in list(climate_lookup.keys())[:3]:
        for m in sorted(climate_lookup[d].keys())[:3]:
            v = climate_lookup[d][m]
            print(f"  {d:20s} month={m:2d}  temp={v['temp']:.1f}°C  "
                  f"humidity={v['humidity']:.1f}%")

    # ── Cross-month training pairs ────────────────────────────────────────────
    # For every (district, current_month) pair with every other month as the
    # harvest target, with temp/humidity perturbations so the model generalises
    # to unseen OpenWeather current values.
    training_rows = []

    for district, group in monthly_avg.groupby("District"):
        group         = group.sort_values("month").reset_index(drop=True)
        if len(group) < 2:
            continue
        district_code = int(group["district_code"].iloc[0])
        dist_climate  = climate_lookup[district]

        for i in range(len(group)):
            cur     = group.iloc[i]
            cm      = int(cur["month"])
            base_ct = float(cur["avg_temp"])       # °C
            base_ch = float(cur["avg_humidity"])   # %

            nct = dist_climate[cm]["temp"]
            nch = dist_climate[cm]["humidity"]

            # Perturbations: ±2°C, ±8% → model generalises to real-time values
            for ts in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for hs in [-8.0, -4.0, 0.0, 4.0, 8.0]:
                    ct = base_ct + ts
                    ch = float(max(0.0, min(100.0, base_ch + hs)))

                    for j in range(len(group)):
                        if i == j:
                            continue
                        fut = group.iloc[j]
                        hm  = int(fut["month"])
                        if hm not in dist_climate:
                            continue
                        nht = dist_climate[hm]["temp"]
                        nhh = dist_climate[hm]["humidity"]

                        training_rows.append({
                            "district_code":           district_code,
                            "current_temp":            ct,
                            "current_humidity":        ch,
                            "current_month":           cm,
                            "harvest_month":           hm,
                            "normal_current_temp":     nct,
                            "normal_current_humidity": nch,
                            "normal_harvest_temp":     nht,
                            "normal_harvest_humidity": nhh,
                            "temp_anomaly":            ct - nct,
                            "humidity_anomaly":        ch - nch,
                            "future_temp":             float(fut["avg_temp"]),
                            "future_humidity":         float(fut["avg_humidity"]),
                        })

    paired_df = pd.DataFrame(training_rows).drop_duplicates().reset_index(drop=True)

    if paired_df.empty:
        raise ValueError(
            "No training pairs created. Check Weather_dataset.csv content."
        )

    print(f"\nTraining pairs shape: {paired_df.shape}")
    print(
        f"future_temp range : "
        f"{paired_df['future_temp'].min():.1f} – {paired_df['future_temp'].max():.1f} °C"
    )
    print(
        f"future_humidity range : "
        f"{paired_df['future_humidity'].min():.1f} – "
        f"{paired_df['future_humidity'].max():.1f} %"
    )

    return paired_df