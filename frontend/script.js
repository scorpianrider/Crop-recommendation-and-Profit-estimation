console.log("Crop Guide Loaded");

/* ============================================================
   DOM REFERENCES
   ============================================================ */
const landInput = document.getElementById("landArea");
const landError = document.getElementById("landError");
const costInput = document.getElementById("costInput");
const costError = document.getElementById("costError");
const formContainer = document.getElementById("formContainer");
const resultContainer = document.getElementById("resultContainer");
const recommendForm = document.getElementById("recommendForm");
const backButton = document.getElementById("backButton");
const formHeader = document.getElementById("formHeader");
const selectCropBtn = document.getElementById("selectCropBtn");
const predictionSection = document.getElementById("predictionSection");

const locationSelected = document.querySelector("#locationSelect .select-selected");
const soilColorSelected = document.querySelector("#soilColorSelect .select-selected");
const soilTypeSelected = document.querySelector("#soilTypeSelect .select-selected");
const plantingSelected = document.querySelector("#plantingMonthSelect .select-selected");
const harvestYearSelected = document.querySelector("#harvestYearSelect .select-selected");
const harvestMonthSelected = document.getElementById("harvestMonthSelected");
const drainageSelected = document.querySelector("#drainageSelect .select-selected");
const waterRetentionSelected = document.querySelector("#waterRetentionSelect .select-selected");

const locationError = document.getElementById("locationError");
const soilColorError = document.getElementById("soilColorError");
const soilTypeError = document.getElementById("soilTypeError");
const plantingMonthError = document.getElementById("plantingMonthError");
const harvestYearError = document.getElementById("harvestYearError");
const harvestMonthError = document.getElementById("harvestMonthError");
const drainageError = document.getElementById("drainageError");
const waterRetentionError = document.getElementById("waterRetentionError");

/* ============================================================
   MONTH MAP
   ============================================================ */
const MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
};

/* ============================================================
   CROP COST PER ACRE
   ============================================================ */
const cropCostPerAcre = {
    "Rice": 38000, "Maize": 25000, "Banana": 56000, "Jute": 30000,
    "Pulses": 20000, "Mango": 30000, "Papaya": 45000, "Tea": 70000,
    "Aloe vera": 20000, "Arecanut": 48000, "Ashwagandha": 18000, "Rose": 60000,
    "Blackgram": 18000, "Chickpea": 20000, "Coconut": 35000, "Coffee": 60000,
    "Cotton": 42000, "Grapes": 80000, "Kidneybeans": 22000, "Mothbeans": 20000,
    "Mungbeans": 18000, "Muskmelon": 56000, "Orange": 50000, "Pigeonpeas": 20000,
    "Pomegranate": 50000, "Watermelon": 50000, "Apple": 70000, "Cabbage": 51500,
    "Cauliflower": 51500, "Green Chillies": 46000, "Carrot": 35000, "Ginger": 60000,
    "Garlic": 45000, "Onion": 50000, "Brinjal": 50000, "Button Mushrooms": 100000,
    "Potato": 45000, "Capsicum": 49000, "Tomato": 61000, "Lady's Finger": 40600,
    "DragonFruit": 70000, "Olive": 60000, "Marigold": 40000, "Beetroot": 30000,
    "Lettuce": 35000, "Corn": 25000, "Green Peas": 28000, "Cucumber": 30000,
    "Guava": 30000, "Turmeric": 45000, "Rajma": 22000, "Pumpkin": 25000,
    "Litchi": 45000, "Broccoli": 45000, "Spinach": 20000, "Groundnut": 25000,
    "Jackfruit": 30000, "Radish": 20000, "Chinese Cabbage": 30000, "Drumstick": 30000,
    "Soybean": 22000, "Sweet Potato": 25000, "Poppy Seeds": 40000, "Coriander": 32000,
    "Walnuts": 60000, "Cashewnuts": 45000, "French Beans": 30000, "Sugarcane": 43000,
    "Bajra": 20000, "Mustard": 22000
};

/* ============================================================
   BUDGET FILTER
   ============================================================ */
function checkBudgetForCrops(crops, farmerBudget, landArea) {
    const validCrops = [];
    crops.forEach(crop => {
        const cost = cropCostPerAcre[crop] || cropCostPerAcre[
            Object.keys(cropCostPerAcre).find(k => k.toLowerCase() === crop.toLowerCase())
        ];
        if (!cost) { validCrops.push(crop); return; }
        if (cost * landArea <= farmerBudget) validCrops.push(crop);
    });
    return validCrops;
}

/* ============================================================
   SHOW CROPS WITH RADIO BUTTONS
   ============================================================ */
function showRecommendedCrops(crops) {
    const cropList = document.getElementById("cropList");
    if (!cropList) return;
    cropList.innerHTML = "";
    if (!crops || crops.length === 0) {
        cropList.innerHTML = `<p style="padding:10px;color:#666;">No crops available within your budget.</p>`;
        return;
    }
    crops.forEach((crop, index) => {
        const label = document.createElement("label");
        label.innerHTML = `
            <input type="radio" name="crop" value="${crop}" ${index === 0 ? "checked" : ""}>
            ${crop}
        `;
        cropList.appendChild(label);
    });
}

/* ============================================================
   WEATHER SOURCE BADGE
   Shows whether weather came from live forecast or ML model
   ============================================================ */
function showWeatherSourceBadge(source) {
    const badge = document.getElementById("weatherSourceBadge");
    if (!badge) return;
    if (source === "open-meteo-forecast") {
        badge.textContent = "📡 Live forecast data (Open-Meteo)";
        badge.style.color = "#1a7a1a";
        badge.style.background = "#e8f5e9";
    } else {
        badge.textContent = "🤖 ML model prediction";
        badge.style.color = "#7a5a00";
        badge.style.background = "#fff8e1";
    }
    badge.style.display = "inline-block";
}

/* ============================================================
   FORM VALIDATION & SUBMIT
   ============================================================ */
if (recommendForm) {
    recommendForm.addEventListener("submit", function (e) {
        e.preventDefault();
        let hasError = false;

        if (!landInput.value.trim() || Number(landInput.value) <= 0) {
            landError.textContent = "Enter valid land area"; hasError = true;
        } else { landError.textContent = ""; }

        if (!costInput.value.trim() || Number(costInput.value) <= 0) {
            costError.textContent = "Enter valid cost"; hasError = true;
        } else { costError.textContent = ""; }

        if (locationSelected.classList.contains("placeholder")) {
            locationError.textContent = "Select location"; hasError = true;
        } else { locationError.textContent = ""; }

        if (soilColorSelected.classList.contains("placeholder")) {
            soilColorError.textContent = "Select soil color"; hasError = true;
        } else { soilColorError.textContent = ""; }

        if (soilTypeSelected.classList.contains("placeholder")) {
            soilTypeError.textContent = "Select soil type"; hasError = true;
        } else { soilTypeError.textContent = ""; }

        if (drainageSelected.classList.contains("placeholder")) {
            drainageError.textContent = "Select drainage level"; hasError = true;
        } else { drainageError.textContent = ""; }

        if (waterRetentionSelected.classList.contains("placeholder")) {
            waterRetentionError.textContent = "Select water retention"; hasError = true;
        } else { waterRetentionError.textContent = ""; }

        if (plantingSelected.classList.contains("placeholder")) {
            plantingMonthError.textContent = "Select planting month"; hasError = true;
        } else { plantingMonthError.textContent = ""; }

        if (harvestYearSelected.classList.contains("placeholder")) {
            harvestYearError.textContent = "Select harvest year"; hasError = true;
        } else { harvestYearError.textContent = ""; }

        if (harvestMonthSelected.classList.contains("placeholder")) {
            harvestMonthError.textContent = "Select harvest month"; hasError = true;
        } else { harvestMonthError.textContent = ""; }

        if (hasError) return;

        formHeader.style.display = "none";
        formContainer.style.display = "none";
        resultContainer.style.display = "block";
        predictionSection.style.display = "none";
        document.getElementById("suggestionBox").style.display = "none";

        getRecommendation();
    });
}

/* ============================================================
   STEP 1 — Get weather forecast / ML prediction → show crops
   ============================================================ */
async function getRecommendation() {
    const district = locationSelected.innerText.trim();
    const harvestMonthTx = harvestMonthSelected.innerText.trim();
    const plantingMonthTx = plantingSelected.innerText.trim();
    const harvestMonth = MONTH_MAP[harvestMonthTx] || 1;
    const plantingMonth = MONTH_MAP[plantingMonthTx] || 1;
    const harvestYear = parseInt(harvestYearSelected.innerText.trim()) || new Date().getFullYear();

    /* Loading state */
    document.getElementById("temperatureResult").innerText = "Loading…";
    document.getElementById("humidityResult").innerText = "Loading…";
    document.getElementById("rainfallResult").innerText = "Loading…";
    const weatherBadge = document.getElementById("weatherSourceBadge");
    if (weatherBadge) { weatherBadge.style.display = "none"; }
    document.getElementById("cropList").innerHTML =
        `<p class="loading-msg">Fetching weather and running models…</p>`;

    try {
        /*
         * POST /predict-crop
         * Backend flow:
         *   1. Open-Meteo 16-day forecast → if harvest month falls in window,
         *      uses real forecast data directly (most accurate).
         *   2. If beyond 16-day window:
         *      OpenWeather current → Weather ML model → Rainfall ML model
         *   3. Crop ML model → recommended crops
         */
        const response = await fetch("http://127.0.0.1:5000/predict-crop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                district: district,
                harvest_month: harvestMonth,
                harvest_year: harvestYear,
                planting_month: plantingMonth
            })
        });

        const data = await response.json();
        console.log("predict-crop response:", data);

        if (!response.ok) throw new Error(data.error || "Failed to get recommendation");

        /* Show predicted weather for harvest month */
        const pw = data.predicted_weather;
        document.getElementById("temperatureResult").innerText = pw.temperature + " °C";
        document.getElementById("humidityResult").innerText = pw.humidity + " %";
        document.getElementById("rainfallResult").innerText = pw.rainfall + " mm";

        /* Show whether we used live forecast or ML */
        showWeatherSourceBadge(data.weather_source);

        /* Budget filter */
        const crops = data.recommended_crops || [];
        const farmerBudget = Number(costInput.value);
        const landArea = Number(landInput.value);
        const afterBudget = checkBudgetForCrops(crops, farmerBudget, landArea);
        showRecommendedCrops(afterBudget.length ? afterBudget : crops);

        if (afterBudget.length === 0 && crops.length > 0) {
            alert("All recommended crops exceed your budget. Showing them anyway — consider adjusting your budget.");
        }

    } catch (error) {
        console.error("predict-crop error:", error);

        /* Fallback */
        document.getElementById("temperatureResult").innerText = "30 °C";
        document.getElementById("humidityResult").innerText = "61 %";
        document.getElementById("rainfallResult").innerText = "5.6 mm";

        const fallbackCrops = ["Rice", "Maize", "Groundnut", "Cotton", "Blackgram", "Tomato", "Cauliflower"];
        const farmerBudget = Number(costInput.value);
        const landArea = Number(landInput.value);
        const result = checkBudgetForCrops(fallbackCrops, farmerBudget, landArea);
        showRecommendedCrops(result.length ? result : fallbackCrops);

        alert("Backend not reachable. Showing sample data. Please start the Flask server.");
    }
}

/* ============================================================
   STEP 2 — User selects crop → call /predict-details
   ============================================================ */
if (selectCropBtn) {
    selectCropBtn.addEventListener("click", async function () {
        const selectedCropInput = document.querySelector('input[name="crop"]:checked');
        if (!selectedCropInput) { alert("Please select a crop"); return; }

        const selectedCrop = selectedCropInput.value;
        const district = locationSelected.innerText.trim();
        const soilColour = soilColorSelected.innerText.trim();
        const soilTexture = soilTypeSelected.innerText.trim();
        const waterRetention = waterRetentionSelected.innerText.trim();
        const drainage = drainageSelected.innerText.trim();
        const areaAcres = Number(landInput.value);
        const budget = Number(costInput.value);
        const harvestMonthTx = harvestMonthSelected.innerText.trim();
        const harvestYearTx = harvestYearSelected.innerText.trim();
        const harvestMonth = MONTH_MAP[harvestMonthTx] || new Date().getMonth() + 1;
        const harvestYear = parseInt(harvestYearTx) || new Date().getFullYear();

        document.getElementById("selectedCropResult").innerText = selectedCrop;
        document.getElementById("irrigationResult").innerText = "Loading…";
        document.getElementById("fertilizerResult").innerText = "Loading…";
        document.getElementById("yieldResult").innerText = "Loading…";
        document.getElementById("marketPriceResult").innerText = "Loading…";
        document.getElementById("profitResult").innerText = "Loading…";
        document.getElementById("suggestionText").innerText = "Loading…";
        predictionSection.style.display = "grid";
        document.getElementById("suggestionBox").style.display = "flex";

        try {
            const resp = await fetch("http://127.0.0.1:5000/predict-details", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: selectedCrop,
                    district: district,
                    soil_colour: soilColour,
                    soil_texture: soilTexture,
                    water_retention: waterRetention,
                    drainage: drainage,
                    area_acres: areaAcres,
                    harvest_month: harvestMonth,
                    harvest_year: harvestYear,
                    budget: budget
                })
            });

            const details = await resp.json();
            console.log("predict-details response:", details);

            if (!resp.ok) throw new Error(details.error || "Details prediction failed");

            document.getElementById("irrigationResult").innerText = details.irrigation || "--";
            document.getElementById("fertilizerResult").innerText = details.fertilizer || "--";
            document.getElementById("yieldResult").innerText = details.predicted_yield || "--";
            document.getElementById("marketPriceResult").innerText = details.market_price || "--";
            document.getElementById("profitResult").innerText = details.estimated_profit || "--";
            document.getElementById("suggestionText").innerText = details.irrigation_reason || "--";

        } catch (err) {
            console.error("predict-details error:", err);

            const dummyDetails = {
                "Rice": { irrigation: "Drip Irrigation", fertilizer: "Urea + Ammonium Sulphate", yield: "28.81 quintals", price: "₹1954 / quintal", profit: "₹25,000", suggestion: "Drip irrigation is best suited for rice given high water retention and moderate drainage." },
                "Maize": { irrigation: "Sprinkler Irrigation", fertilizer: "NPK 20:20:0 + Urea", yield: "22 quintals", price: "₹2,100 / quintal", profit: "₹21,200", suggestion: "Sprinkler irrigation suits maize well in well-drained soil." },
                "Groundnut": { irrigation: "Drip Irrigation", fertilizer: "Gypsum + SSP + Urea", yield: "16 quintals", price: "₹6,000 / quintal", profit: "₹46,000", suggestion: "Drip irrigation is ideal for groundnut in sandy or loamy soil." },
                "Cotton": { irrigation: "Drip Irrigation", fertilizer: "NPK + Micronutrients", yield: "10 quintals", price: "₹7,200 / quintal", profit: "₹30,000", suggestion: "Drip irrigation performs well for cotton in black soil." },
                "Tomato": { irrigation: "Drip Irrigation", fertilizer: "NPK + Calcium Nitrate", yield: "80 quintals", price: "₹1,500 / quintal", profit: "₹55,000", suggestion: "Drip irrigation is optimal for tomato cultivation." },
                "Cauliflower": { irrigation: "Sprinkler Irrigation", fertilizer: "NPK + Boron", yield: "60 quintals", price: "₹1,200 / quintal", profit: "₹30,000", suggestion: "Sprinkler irrigation suits cauliflower in loamy soil." },
            };
            const d = dummyDetails[selectedCrop] || dummyDetails["Rice"];
            document.getElementById("irrigationResult").innerText = d.irrigation;
            document.getElementById("fertilizerResult").innerText = d.fertilizer;
            document.getElementById("yieldResult").innerText = d.yield;
            document.getElementById("marketPriceResult").innerText = d.price;
            document.getElementById("profitResult").innerText = d.profit;
            document.getElementById("suggestionText").innerText = d.suggestion;

            alert("Backend not reachable. Showing sample data for selected crop.");
        }
    });
}

/* ============================================================
   BACK BUTTON
   ============================================================ */
if (backButton) {
    backButton.addEventListener("click", () => {
        resultContainer.style.display = "none";
        formContainer.style.display = "block";
        formHeader.style.display = "block";
    });
}

/* ============================================================
   INPUT ERROR CLEAR
   ============================================================ */
if (landInput) landInput.addEventListener("input", () => landError.textContent = "");
if (costInput) costInput.addEventListener("input", () => costError.textContent = "");

/* ============================================================
   CUSTOM DROPDOWN SYSTEM
   ============================================================ */
function setupCustomSelect(selectDiv) {
    const selected = selectDiv.querySelector(".select-selected");
    const options = selectDiv.querySelectorAll(".select-options li");

    selected.addEventListener("click", function (e) {
        e.stopPropagation();
        document.querySelectorAll(".custom-select").forEach(s => {
            if (s !== selectDiv) s.classList.remove("active");
        });
        selectDiv.classList.toggle("active");
    });

    options.forEach(option => {
        option.addEventListener("click", function (e) {
            e.stopPropagation();
            if (option.dataset.disabled === "true") return;
            selected.textContent = option.textContent;
            selected.classList.remove("placeholder");
            selected.style.color = "#000";
            selectDiv.classList.remove("active");
        });
    });
}

document.addEventListener("click", function () {
    document.querySelectorAll(".custom-select").forEach(s => s.classList.remove("active"));
});

/* ============================================================
   HARVEST MONTH / YEAR LOGIC
   ============================================================ */
document.addEventListener("DOMContentLoaded", function () {
    const monthNames = Object.keys(MONTH_MAP);
    const currentYear = new Date().getFullYear();
    const customSelects = document.querySelectorAll(".custom-select");

    customSelects.forEach(s => setupCustomSelect(s));

    const plantingSelect = document.getElementById("plantingMonthSelect");
    const harvestYearDiv = document.getElementById("harvestYearSelect");
    const harvestYearSelLocal = harvestYearDiv ? harvestYearDiv.querySelector(".select-selected") : null;
    const harvestYearOpts = document.getElementById("harvestYearOptions");
    const harvestMonthSelLocal = document.getElementById("harvestMonthSelected");
    const harvestMonthOpts = document.getElementById("harvestMonthOptions");

    if (!harvestYearOpts || !harvestMonthOpts) return;

    [currentYear, currentYear + 1].forEach(year => {
        const li = document.createElement("li");
        li.textContent = year;
        li.addEventListener("click", function (e) {
            e.stopPropagation();
            harvestYearSelLocal.textContent = year;
            harvestYearSelLocal.classList.remove("placeholder");
            harvestYearSelLocal.style.color = "#000";
            harvestYearDiv.classList.remove("active");

            const plantingOpts = plantingSelect.querySelectorAll(".select-options li");
            let plantingIndex = -1;
            plantingOpts.forEach((opt, i) => {
                if (opt.textContent.trim() === plantingSelect.querySelector(".select-selected").textContent.trim()) {
                    plantingIndex = i;
                }
            });
            if (plantingIndex !== -1) populateHarvestMonths(plantingIndex);
        });
        harvestYearOpts.appendChild(li);
    });

    function populateHarvestMonths(plantingIndex) {
        harvestMonthOpts.innerHTML = "";
        const selectedYear = parseInt(harvestYearSelLocal.textContent);

        monthNames.forEach((month, index) => {
            const li = document.createElement("li");
            li.textContent = month;

            if (selectedYear === currentYear && index <= plantingIndex) {
                li.style.color = "#999";
                li.style.cursor = "not-allowed";
                li.dataset.disabled = "true";
            }

            li.addEventListener("click", function (e) {
                e.stopPropagation();
                if (li.dataset.disabled === "true") return;
                harvestMonthSelLocal.textContent = month;
                harvestMonthSelLocal.classList.remove("placeholder");
                harvestMonthSelLocal.style.color = "#000";
                harvestMonthSelLocal.parentElement.classList.remove("active");
            });

            harvestMonthOpts.appendChild(li);
        });
    }

    if (plantingSelect) {
        plantingSelect.querySelectorAll(".select-options li").forEach((option, index) => {
            option.addEventListener("click", function () {
                harvestYearSelLocal.textContent = currentYear;
                harvestYearSelLocal.classList.remove("placeholder");
                harvestYearSelLocal.style.color = "#000";
                populateHarvestMonths(index);
            });
        });
    }
});