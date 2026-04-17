# utils/growth_filter.py

import os
import pandas as pd


def filter_by_growth_period(crops, planting_month, harvest_month):
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    csv_path     = os.path.join(datasets_dir, "crop_growth_period.csv")

    if not os.path.exists(csv_path):
        # If file missing, return all crops unfiltered
        return crops

    growth_data = pd.read_csv(csv_path)

    month_diff = harvest_month - planting_month
    if month_diff <= 0:
        month_diff += 12
    duration_days = month_diff * 30

    suitable = []
    for crop in crops:
        row = growth_data[growth_data["crop"].str.lower() == crop.lower()]
        if row.empty:
            # Unknown crop — include it (don't block it)
            suitable.append(crop)
            continue
        min_days = row.iloc[0]["min_days"]
        max_days = row.iloc[0]["max_days"]
        if min_days <= duration_days <= max_days:
            suitable.append(crop)

    return suitable
