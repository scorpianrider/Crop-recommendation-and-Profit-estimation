# preprocessing/rainfall_preprocessing.py
#
# FIXES:
#   1. CSV rows are DAILY rainfall in mm — monthly total = SUM (not mean) of
#      daily values per district-month group. Previously using mean() gave
#      mm/day averages instead of monthly totals, producing ~30x underestimates.
#   2. Cross-month training pairs (current_month ≠ harvest_month) so the
#      model learns to predict a FUTURE month's rainfall.
#   3. Target column renamed to "future_rainfall_mm" (was "future_rainfall_cm"
#      in train_rainfall_model.py — mismatch caused KeyError at training time).
#   4. Perturbations added so the model generalises to unseen API values.
#   5. No cm↔mm conversion: CSV values are already in mm.

import pandas as pd
import os
import joblib

DISTRICT_ENCODER: dict = {}


def preprocess_rainfall_data():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    base_dir     = os.path.join(current_dir, "..")
    datasets_dir = os.path.join(base_dir, "..", "datasets")
    models_dir   = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    files = [
        "rainfall_by_districts_2023.csv",
        "rainfall_by_districts_2024.csv",
        "rainfall_by_districts_2025.csv",
    ]

    df_list = []
    for f in files:
        path = os.path.join(datasets_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        print(f"Loading: {path}")
        df_list.append(pd.read_csv(path))

    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()

    for col in ["District", "Month", "Year", "Avg_rainfall"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["Month"]        = pd.to_numeric(df["Month"],        errors="coerce")
    df["Year"]         = pd.to_numeric(df["Year"],         errors="coerce")
    df["Avg_rainfall"] = pd.to_numeric(df["Avg_rainfall"], errors="coerce")
    df = df.dropna(subset=["District", "Month", "Avg_rainfall"])

    # ── FIX: CSV has DAILY rows in mm — SUM to get monthly total rainfall ────
    # Do NOT use mean() here; daily rows must be summed per district-month-year.
    monthly_total_df = (
        df.groupby(["District", "Year", "Month"])["Avg_rainfall"]
        .sum()
        .reset_index()
        .rename(columns={"Avg_rainfall": "monthly_rainfall_mm"})
    )

    rain_mean = monthly_total_df["monthly_rainfall_mm"].mean()
    print(f"  Monthly total rainfall mean = {rain_mean:.1f} mm")

    DISTRICT_ALIASES = {
        "Tiruvallur":  "Thiruvallur",
        "Villupuram":  "Viluppuram",
        "Kanchipuram": "Kancheepuram",
    }
    monthly_total_df["District"] = monthly_total_df["District"].apply(
        lambda x: DISTRICT_ALIASES.get(str(x).strip(), str(x).strip())
    )

    # ── District encoder ──────────────────────────────────────────────────────
    unique_districts = sorted(monthly_total_df["District"].unique())
    for i, d in enumerate(unique_districts):
        DISTRICT_ENCODER[d] = i
    monthly_total_df["district_code"] = monthly_total_df["District"].map(DISTRICT_ENCODER)

    joblib.dump(DISTRICT_ENCODER, os.path.join(models_dir, "rainfall_district_encoder.pkl"))
    print(f"Saved rainfall_district_encoder with {len(DISTRICT_ENCODER)} districts")

    # ── Monthly climate averages (across years) ───────────────────────────────
    monthly_avg = (
        monthly_total_df.groupby(["District", "Month"])["monthly_rainfall_mm"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_rainfall_mm": "monthly_avg_rainfall_mm"})
    )

    climate_lookup: dict = {}
    for _, row in monthly_avg.iterrows():
        d = row["District"]
        m = int(row["Month"])
        climate_lookup.setdefault(d, {})[m] = float(row["monthly_avg_rainfall_mm"])

    joblib.dump(climate_lookup, os.path.join(models_dir, "rainfall_monthly_climate.pkl"))
    print("Saved rainfall_monthly_climate.pkl")

    print("\nMonthly rainfall climate averages sample (mm):")
    print(monthly_avg.head(8).to_string(index=False))

    # ── Cross-month training pairs ────────────────────────────────────────────
    training_rows = []

    for (district, year), year_group in monthly_total_df.groupby(["District", "Year"]):
        year_group    = year_group.sort_values("Month").reset_index(drop=True)
        if len(year_group) < 2:
            continue

        district_code = int(DISTRICT_ENCODER.get(district, -1))
        if district_code == -1:
            continue

        dist_climate = climate_lookup.get(district, {})

        for i, cur_row in year_group.iterrows():
            cm        = int(cur_row["Month"])
            cr_mm     = float(cur_row["monthly_rainfall_mm"])
            normal_cr = dist_climate.get(cm, cr_mm)
            anomaly   = cr_mm - normal_cr

            # Perturbations so model generalises to unseen API values (±15 mm)
            for delta in [-15.0, -7.0, 0.0, 7.0, 15.0]:
                perturbed_cr      = max(0.0, cr_mm + delta)
                perturbed_anomaly = perturbed_cr - normal_cr

                for j, fut_row in year_group.iterrows():
                    if i == j:
                        continue
                    hm        = int(fut_row["Month"])
                    future_mm = float(fut_row["monthly_rainfall_mm"])
                    normal_hr = dist_climate.get(hm, future_mm)

                    training_rows.append({
                        "district_code":           district_code,
                        "current_rainfall":        perturbed_cr,
                        "current_month":           cm,
                        "harvest_month":           hm,
                        "normal_current_rainfall": normal_cr,
                        "normal_harvest_rainfall": normal_hr,
                        "rainfall_anomaly":        perturbed_anomaly,
                        "future_rainfall_mm":      future_mm,   # target — always mm
                    })

    paired_df = pd.DataFrame(training_rows).drop_duplicates().reset_index(drop=True)
    if paired_df.empty:
        raise ValueError("No training pairs created. Check rainfall CSVs.")

    print(f"\nRainfall training pairs shape: {paired_df.shape}")
    print(f"future_rainfall_mm range: "
          f"{paired_df['future_rainfall_mm'].min():.1f} – "
          f"{paired_df['future_rainfall_mm'].max():.1f} mm")
    return paired_df