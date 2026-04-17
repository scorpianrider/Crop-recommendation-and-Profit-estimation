# data/merge_market_data.py
# Run once:  python -m data.merge_market_data

import os
import pandas as pd

current_dir  = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(current_dir, "..", "..", "datasets")
folder       = os.path.join(datasets_dir, "crop_prices")
out_path     = os.path.join(datasets_dir, "combined_price_data.csv")


def merge_market_data():
    if not os.path.exists(folder):
        raise FileNotFoundError(f"crop_prices folder not found: {folder}")

    files    = [f for f in os.listdir(folder) if f.endswith(".csv")]
    all_data = []

    for file in files:
        path = os.path.join(folder, file)
        try:
            df = pd.read_csv(path, header=1)
        except Exception as e:
            print(f"  Skipping {file}: {e}")
            continue

        price_col = [col for col in df.columns if "Modal Price" in col]
        if not price_col:
            print(f"  No 'Modal Price' column in {file} — skipping")
            continue

        price_col = price_col[0]
        df = df[["Commodity", "Date", price_col]].rename(columns={
            "Commodity": "crop",
            "Date":      "date",
            price_col:   "price"
        })

        df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna()
        all_data.append(df)
        print(f"  Loaded: {file}  ({len(df)} rows)")

    if not all_data:
        raise ValueError("No valid CSV files found in crop_prices/")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["crop", "date"])
    combined.to_csv(out_path, index=False)
    print(f"\nMerged {len(combined)} rows → {out_path}")


if __name__ == "__main__":
    merge_market_data()
