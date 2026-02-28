import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------

st.set_page_config(page_title="AgriMarket Twin", layout="wide")

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #e8f5e9;
}

/* Headings */
h1 { color: #145a32 !important; font-weight: 700; }
h2, h3 { color: #1b5e20 !important; }

/* Labels */
label {
    color: #145a32 !important;
    font-weight: 600 !important;
}

/* Metric styling */
div[data-testid="stMetricLabel"] {
    color: #145a32 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #0b3d0b !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #2f2f2f !important;
    border: 2px solid #2e7d32 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

div[role="listbox"] {
    background-color: #2f2f2f !important;
}

div[role="option"] {
    background-color: #2f2f2f !important;
    color: white !important;
}

div[role="option"]:hover {
    background-color: #2e7d32 !important;
}

/* Buttons */
.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    border-radius: 10px !important;
    height: 3em !important;
    font-weight: 600 !important;
}

.stButton > button * {
    color: white !important;
}

.stButton > button:hover {
    background-color: #145a32 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriMarket Twin – Decision Intelligence Engine")

# ----------------------------------
# LOAD DATA
# ----------------------------------

pipeline = joblib.load("agri_yield.pkl")
model_data = pd.read_csv("crop.csv")
soil_data = pd.read_csv("soil_dataset.csv")
weather_data = pd.read_csv("weather_dataset.csv")

soil_data.columns = soil_data.columns.str.strip()
weather_data.columns = weather_data.columns.str.strip()

# ----------------------------------
# SESSION STATE
# ----------------------------------

if "screen" not in st.session_state:
    st.session_state.screen = "context"

if "step" not in st.session_state:
    st.session_state.step = 0

if "shock_values" not in st.session_state:
    st.session_state.shock_values = {}

# ----------------------------------
# STEP 1 – INPUT + BASELINE PREDICTION
# ----------------------------------

if st.session_state.screen == "context":

    st.subheader("📍 Farmer Information")

    state = st.selectbox("Select State", sorted(soil_data["State"].unique()))
    region = st.selectbox(
        "Select Region",
        sorted(soil_data[soil_data["State"] == state]["Region"].unique())
    )

    crop = st.selectbox("Crop Type", sorted(model_data["crop"].unique()))
    fertilizer = st.selectbox("Fertilizer Type", sorted(model_data["fertilizer"].unique()))

    area = st.number_input("Area (Hectares)", min_value=1, step=1)
    price = st.number_input("Market Price (₹ per kg)", min_value=1, step=1)

    soil_filtered = soil_data[(soil_data["State"] == state) &
                              (soil_data["Region"] == region)]
    weather_filtered = weather_data[(weather_data["State"] == state) &
                                    (weather_data["Region"] == region)]

    if not soil_filtered.empty:
        N = int(soil_filtered["N"].mean())
        P = int(soil_filtered["P"].mean())
        K = int(soil_filtered["K"].mean())
        ph = float(round(soil_filtered["pH"].mean(), 1))
    else:
        N, P, K, ph = 90, 40, 40, 6.5

    if not weather_filtered.empty:
        temperature = int(weather_filtered["Temperature"].mean())
        humidity = int(weather_filtered["Humidity"].mean())
        rainfall = int(weather_filtered["Rainfall"].mean())
    else:
        temperature, humidity, rainfall = 25, 70, 200

    st.subheader("🌱 Baseline Conditions (Editable)")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", value=N, step=1)
        P = st.number_input("Phosphorus (P)", value=P, step=1)
        K = st.number_input("Potassium (K)", value=K, step=1)

    with col2:
        temperature = st.number_input("Temperature (°C)", value=temperature, step=1)
        humidity = st.number_input("Humidity (%)", value=humidity, step=1)

    with col3:
        rainfall = st.number_input("Rainfall (mm)", value=rainfall, step=1)
        ph = st.number_input("Soil pH", value=ph, step=0.1)

    if st.button("📊 Predict Baseline Yield"):

        base_input = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "crop": crop,
            "fertilizer": fertilizer
        }])

        baseline_yield = pipeline.predict(base_input)[0]

        st.session_state.base_input = base_input
        st.session_state.baseline_yield = baseline_yield
        st.session_state.area = area
        st.session_state.price = price
        st.session_state.baseline_revenue = baseline_yield * area * price

        st.session_state.screen = "baseline_result"
        st.rerun()

# ----------------------------------
# STEP 2 – SHOW BASELINE RESULT
# ----------------------------------

elif st.session_state.screen == "baseline_result":

    st.subheader("📊 Baseline Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Yield (per hectare)",
                round(st.session_state.baseline_yield,2))

    col2.metric("Baseline Production",
                round(st.session_state.baseline_yield * st.session_state.area,2))

    col3.metric("Baseline Revenue (₹)",
                round(st.session_state.baseline_revenue,2))

    if st.button("🚀 Proceed to Scenario Simulation"):
        st.session_state.screen = "tiles"
        st.rerun()

# ----------------------------------
# STEP 3 – TILE FLOW (UNCHANGED)
# ----------------------------------

elif st.session_state.screen == "tiles":

    variables = ["Temperature", "Humidity", "Rainfall", "Soil pH"]

    scenario_map = {
        "Temperature": [-3, -1, 0, 1, 3],
        "Humidity": [-20, -10, 0, 10, 20],
        "Rainfall": [-30, -15, 0, 15, 30],
        "Soil pH": [-1, -0.5, 0, 0.5, 1]
    }

    unit_map = {
        "Temperature": "°C",
        "Humidity": "%",
        "Rainfall": "%",
        "Soil pH": "pH units"
    }

    current_var = variables[st.session_state.step]
    st.subheader(f"🧩 Select {current_var} Scenario")

    options = scenario_map[current_var]
    cols = st.columns(5)

    for i, value in enumerate(options):
        unit = unit_map[current_var]
        label = f"{value} {unit}" if unit != "%" else f"{value}%"

        if cols[i].button(label):
            st.session_state.shock_values[current_var] = value

    if current_var in st.session_state.shock_values:
        st.info(f"Selected Change: {st.session_state.shock_values[current_var]} {unit_map[current_var]}")

    if st.button("Next ➡"):
        if current_var not in st.session_state.shock_values:
            st.warning("Select a scenario first.")
        else:
            if st.session_state.step < len(variables) - 1:
                st.session_state.step += 1
            else:
                st.session_state.screen = "simulate"
            st.rerun()

# ----------------------------------
# STEP 4 – SIMULATION
# ----------------------------------

elif st.session_state.screen == "simulate":

    st.subheader("⚙ Running Monte Carlo Simulation")

    results = []

    for _ in range(1000):
        simulated = st.session_state.base_input.copy()

        simulated["temperature"] += st.session_state.shock_values["Temperature"]
        simulated["humidity"] *= (1 + st.session_state.shock_values["Humidity"]/100)
        simulated["rainfall"] *= (1 + st.session_state.shock_values["Rainfall"]/100)
        simulated["ph"] += st.session_state.shock_values["Soil pH"]

        pred = pipeline.predict(simulated)[0]
        results.append(pred)

    results = np.array(results)

    mean_yield = np.mean(results)
    total_revenue = mean_yield * st.session_state.area * st.session_state.price

    st.session_state.simulated_yield = mean_yield
    st.session_state.simulated_revenue = total_revenue

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated Yield", round(mean_yield,2))
    col2.metric("Simulated Production",
                round(mean_yield * st.session_state.area,2))
    col3.metric("Simulated Revenue (₹)",
                round(total_revenue,2))

    if st.button("📈 View Comparison"):
        st.session_state.screen = "comparison"
        st.rerun()

# ----------------------------------
# STEP 5 – COMPARISON
# ----------------------------------

elif st.session_state.screen == "comparison":

    st.subheader("📊 Baseline vs Scenario Comparison")

    yield_change = st.session_state.simulated_yield - st.session_state.baseline_yield
    revenue_change = st.session_state.simulated_revenue - st.session_state.baseline_revenue

    yield_percent = (yield_change / st.session_state.baseline_yield) * 100
    revenue_percent = (revenue_change / st.session_state.baseline_revenue) * 100

    col1, col2 = st.columns(2)

    col1.metric("Yield Change",
                f"{round(st.session_state.simulated_yield,2)}",
                f"{round(yield_change,2)} ({round(yield_percent,2)}%)")

    col2.metric("Revenue Change (₹)",
                f"{round(st.session_state.simulated_revenue,2)}",
                f"{round(revenue_change,2)} ({round(revenue_percent,2)}%)")

    if st.button("🔄 Restart Simulation"):
        st.session_state.screen = "context"
        st.session_state.step = 0
        st.session_state.shock_values = {}
        st.rerun()
