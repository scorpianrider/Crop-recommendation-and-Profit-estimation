========================================================
  CROP GUIDE — Complete Project
  Crop Recommendation & Yield Prediction using ML
========================================================

FOLDER STRUCTURE
----------------
crop_guide/
├── frontend/               ← Open index.html in browser
│   ├── index.html
│   ├── recommend.html
│   ├── about.html
│   ├── style.css
│   ├── script.js
│   └── farmer_bg.png       ← PLACE YOUR BACKGROUND IMAGE HERE
│
├── backend/                ← Python Flask server
│   ├── app.py              ← Main server (run this)
│   ├── requirements.txt
│   ├── config/
│   ├── preprocessing/
│   ├── training/
│   ├── prediction/
│   ├── utils/
│   ├── data/
│   └── models/             ← .pkl files appear here after training
│
├── datasets/               ← PLACE ALL YOUR CSV FILES HERE
│   ├── Crop recommendation data.csv
│   ├── Fertilizer_data.csv
│   ├── Yield_data.csv
│   ├── Weather_dataset.csv
│   ├── crop_growth_period.csv
│   ├── rainfall_by_districts_2023.csv
│   ├── rainfall_by_districts_2024.csv
│   ├── rainfall_by_districts_2025.csv
│   ├── combined_price_data.csv   ← created by step 3 below
│   ├── irrigation_data.csv       ← created by step 2 below
│   └── crop_prices/              ← PLACE ALL CROP PRICE CSVs HERE
│       ├── Rice.csv
│       ├── Tomato.csv
│       └── ... (all other crop CSVs)
│
└── README.txt (this file)


========================================================
  STEP-BY-STEP SETUP
========================================================

STEP 1 — Place your image
--------------------------
Copy your farmer background image into:
    frontend/farmer_bg.png

(It was named  eb1aaf77-4505-468c-9120-e8da12bdccf6.png  — just rename
it to  farmer_bg.png  and drop it in the frontend/ folder.)


STEP 2 — Place your datasets
------------------------------
Copy ALL your CSV files into the  datasets/  folder.
Copy all your individual crop price CSVs into  datasets/crop_prices/

The file names must match exactly:
  - Crop recommendation data.csv
  - Fertilizer_data.csv
  - Yield_data.csv
  - Weather_dataset.csv
  - crop_growth_period.csv
  - rainfall_by_districts_2023.csv
  - rainfall_by_districts_2024.csv
  - rainfall_by_districts_2025.csv


STEP 3 — Install Python dependencies
--------------------------------------
Open a terminal, go into the backend folder:

    cd crop_guide/backend
    pip install -r requirements.txt


STEP 4 — Generate the irrigation dataset (run ONCE)
-----------------------------------------------------
    cd crop_guide/backend
    python -m data.generate_irrigation_dataset

This creates:  datasets/irrigation_data.csv


STEP 5 — Merge market price CSVs (run ONCE)
---------------------------------------------
    python -m data.merge_market_data

This reads all files from  datasets/crop_prices/
and creates:  datasets/combined_price_data.csv


STEP 6 — Train ALL models (run ONCE, takes a few minutes)
-----------------------------------------------------------
Run these commands one by one from inside  crop_guide/backend/:

    python -m training.train_weather_model
    python -m training.train_rainfall_model
    python -m training.train_crop_model
    python -m training.train_irrigation_model
    python -m training.train_fertilizer_model
    python -m training.train_yield_model
    python -m training.train_market_model

After training, the  backend/models/  folder will contain all .pkl files.
You only need to train once. After that, just run the server.


STEP 7 — Start the Flask server
---------------------------------
    cd crop_guide/backend
    python app.py

You should see:
    Loading models...
    All models loaded successfully.
    * Running on http://127.0.0.1:5000


STEP 8 — Open the website
---------------------------
Open  crop_guide/frontend/index.html  in your browser.
(Or use VS Code Live Server — right-click index.html → Open with Live Server)

The website will talk to http://127.0.0.1:5000 automatically.


========================================================
  DATA FLOW (how the models connect)
========================================================

User fills form → clicks "Get Recommendation"
    │
    ▼
OpenWeather API
    Gives: current temperature, humidity, rainfall (this month)
    │
    ├──► current temp + humidity ──► Weather ML model
    │                                    └──► predicted temp & humidity (at harvest time)
    │
    └──► current rainfall ──────► Rainfall ML model
                                      └──► predicted rainfall (at harvest time)
    │
    ▼
Crop ML model
    Inputs: predicted temp + humidity + rainfall
    Filtered by: district map + growth period + budget (budget in JS)
    Output: list of recommended crops shown on website
    │
User selects a crop → clicks "Select Crop"
    │
    ├──► Irrigation model   → irrigation type + LIME explanation
    ├──► Fertilizer model   → fertilizer name
    ├──► Yield model        → predicted yield (quintals)
    └──► Market Price model → predicted price (₹/quintal)
                                    │
                            Profit = (yield × price) − budget


========================================================
  TROUBLESHOOTING
========================================================

"Model not found" error when starting server:
  → You haven't trained the models yet. Run all training scripts (Step 6).

"District not supported" error:
  → The district name from frontend doesn't match ALLOWED_DISTRICTS.
    Check config/allowed_districts.py

API returns dummy data in browser:
  → The Flask server is not running. Start it with: python app.py

"combined_price_data.csv not found":
  → Run Step 5: python -m data.merge_market_data

Weather API fails:
  → Check your internet connection.
  → The OpenWeather API key is in app.py (OPENWEATHER_API_KEY).
    If it stops working, get a free key at https://openweathermap.org/api
