# data/generate_irrigation_dataset.py
# Run once:  python -m data.generate_irrigation_dataset

import os
import pandas as pd

CROP_PRIMARY_IRRIGATION = {
    "Rice":             "Flood Irrigation",
    "Sugarcane":        "Furrow Irrigation",
    "Apple":            "Drip Irrigation",
    "Mango":            "Basin Irrigation",
    "Banana":           "Drip Irrigation",
    "Coconut":          "Basin Irrigation",
    "Guava":            "Basin Irrigation",
    "Jackfruit":        "Basin Irrigation",
    "Orange":           "Drip Irrigation",
    "Pomegranate":      "Drip Irrigation",
    "Papaya":           "Drip Irrigation",
    "Grapes":           "Drip Irrigation",
    "Arecanut":         "Basin Irrigation",
    "Cashewnuts":       "Basin Irrigation",
    "Tomato":           "Drip Irrigation",
    "Brinjal":          "Drip Irrigation",
    "Cabbage":          "Sprinkler Irrigation",
    "Cauliflower":      "Sprinkler Irrigation",
    "Lady's Finger":    "Drip Irrigation",
    "Green Chillies":   "Drip Irrigation",
    "Capsicum":         "Drip Irrigation",
    "Cucumber":         "Drip Irrigation",
    "Pumpkin":          "Furrow Irrigation",
    "Watermelon":       "Drip Irrigation",
    "Muskmelon":        "Drip Irrigation",
    "Beetroot":         "Sprinkler Irrigation",
    "Carrot":           "Sprinkler Irrigation",
    "Radish":           "Sprinkler Irrigation",
    "Onion":            "Drip Irrigation",
    "Garlic":           "Drip Irrigation",
    "Potato":           "Sprinkler Irrigation",
    "Sweet Potato":     "Sprinkler Irrigation",
    "Green Peas":       "Sprinkler Irrigation",
    "French Beans":     "Sprinkler Irrigation",
    "Maize":            "Sprinkler Irrigation",
    "Bajra":            "Sprinkler Irrigation",
    "Cotton":           "Drip Irrigation",
    "Groundnut":        "Sprinkler Irrigation",
    "Soybean":          "Sprinkler Irrigation",
    "Blackgram":        "Sprinkler Irrigation",
    "Mungbeans":        "Sprinkler Irrigation",
    "Coriander":        "Sprinkler Irrigation",
    "Turmeric":         "Furrow Irrigation",
    "Ginger":           "Sprinkler Irrigation",
    "Drumstick":        "Drip Irrigation",
    "Marigold":         "Drip Irrigation",
    "Rose":             "Drip Irrigation",
    "Button Mushrooms": "Sprinkler Irrigation",
}


def apply_soil_rules(primary, water_retention, drainage):
    wr = water_retention.lower()
    dr = drainage.lower()

    if primary == "Flood Irrigation":
        if dr == "high" or wr == "low":
            return "Furrow Irrigation"
        return "Flood Irrigation"
    elif primary == "Furrow Irrigation":
        if wr == "low":
            return "Drip Irrigation"
        return "Furrow Irrigation"
    elif primary == "Basin Irrigation":
        if wr == "low" or dr == "high":
            return "Drip Irrigation"
        return "Basin Irrigation"
    elif primary == "Drip Irrigation":
        return "Drip Irrigation"
    elif primary == "Sprinkler Irrigation":
        if wr == "low" and dr == "high":
            return "Drip Irrigation"
        return "Sprinkler Irrigation"
    return primary


def generate_dataset():
    rows = []
    for crop, primary in CROP_PRIMARY_IRRIGATION.items():
        for wr in ["High", "Moderate", "Low"]:
            for dr in ["High", "Moderate", "Low"]:
                final = apply_soil_rules(primary, wr, dr)
                rows.append({
                    "Crop":            crop,
                    "Water_Retention": wr,
                    "Drainage":        dr,
                    "Irrigation_Type": final,
                })

    df = pd.DataFrame(rows)

    # Save into datasets/ folder
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    out_path = os.path.join(datasets_dir, "irrigation_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Irrigation dataset saved: {out_path}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    generate_dataset()
