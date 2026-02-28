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
.stApp { background-color: #e8f5e9; }

h1 { color: #145a32 !important; font-weight: 700; }
h2, h3 { color: #1b5e20 !important; }

label { color: #145a32 !important; font-weight: 600 !important; }

div[data-testid="stMetricLabel"] {
    color: #145a32 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #0b3d0b !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    background-color: #2f2f2f !important;
    border: 2px solid #2e7d32 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span { color: white !important; }

div[role="listbox"] { background-color: #2f2f2f !important; }
div[role="option"] { background-color: #2f2f2f !important; color: white !important; }
div[role="option"]:hover { background-color: #2e7d32 !important; }

.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    border-radius: 10px !important;
    height: 3em !important;
    font-weight: 600 !important;
}

.stButton > button * { color: white !important; }
.stButton > button:hover { background-color: #145a32 !important; }
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
# SESSION STATE INIT
# ----------------------------------

if "screen" not in st.session_state:
    st.session_state.screen = "context"

if "shock_values" not in st.session_state:
    st.session_state.shock_values = {
        "Temperature": 0,
        "Humidity": 0,
        "Rainfall": 0,
        "Soil pH": 0
    }

# ----------------------------------
# STEP 1 – BASELINE INPUT
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

    if st.button("📊 Predict Baseline Yield", key="baseline_btn"):

        base_input = pd.DataFrame([{
            "N": N, "P": P, "K": K,
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
        st.session_state.baseline_revenue = baseline_yield * area * price
        st.session_state.area = area
        st.session_state.price = price

        st.session_state.screen = "baseline_result"
        st.rerun()

# ----------------------------------
# STEP 2 – BASELINE RESULT
# ----------------------------------

elif st.session_state.screen == "baseline_result":

    st.subheader("📊 Baseline Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Yield", round(st.session_state.baseline_yield, 2))
    col2.metric("Baseline Production",
                round(st.session_state.baseline_yield * st.session_state.area, 2))
    col3.metric("Baseline Revenue (₹)",
                round(st.session_state.baseline_revenue, 2))

    if st.button("🚀 Proceed to Scenario Simulation", key="to_tiles"):
        st.session_state.screen = "tiles"
        st.rerun()

# ----------------------------------
# STEP 3 – SCENARIO SELECTION
# ----------------------------------

elif st.session_state.screen == "tiles":

    st.subheader("🧩 Scenario Selection")

    # ---------------- TEMPERATURE ----------------
    st.markdown("### 🌡 Temperature Change (°C)")

    temp_options = [-3, -1, 0, 1, 3]
    cols = st.columns(5)

    for i, val in enumerate(temp_options):
        if cols[i].button(f"{val} °C", key=f"temp_{i}"):
            st.session_state.shock_values["Temperature"] = val

    temp_custom = st.number_input(
        "Custom Temperature Change",
        value=int(st.session_state.shock_values.get("Temperature", 0)),
        step=1,
        key="custom_temp"
    )

    st.session_state.shock_values["Temperature"] = temp_custom
    st.success(f"Selected Temperature Change: {temp_custom} °C")

    # ---------------- HUMIDITY ----------------
    st.markdown("### 💧 Humidity Change (%)")

    hum_options = [-20, -10, 0, 10, 20]
    cols = st.columns(5)

    for i, val in enumerate(hum_options):
        if cols[i].button(f"{val}%", key=f"hum_{i}"):
            st.session_state.shock_values["Humidity"] = val

    hum_custom = st.number_input(
        "Custom Humidity Change",
        value=int(st.session_state.shock_values.get("Humidity", 0)),
        step=1,
        key="custom_hum"
    )

    st.session_state.shock_values["Humidity"] = hum_custom
    st.success(f"Selected Humidity Change: {hum_custom} %")

    # ---------------- RAINFALL ----------------
    st.markdown("### 🌧 Rainfall Change (%)")

    rain_options = [-30, -15, 0, 15, 30]
    cols = st.columns(5)

    for i, val in enumerate(rain_options):
        if cols[i].button(f"{val}%", key=f"rain_{i}"):
            st.session_state.shock_values["Rainfall"] = val

    rain_custom = st.number_input(
        "Custom Rainfall Change",
        value=int(st.session_state.shock_values.get("Rainfall", 0)),
        step=1,
        key="custom_rain"
    )

    st.session_state.shock_values["Rainfall"] = rain_custom
    st.success(f"Selected Rainfall Change: {rain_custom} %")

    # ---------------- SOIL PH ----------------
    st.markdown("### 🧪 Soil pH Change")

    ph_options = [-1, -0.5, 0, 0.5, 1]
    cols = st.columns(5)

    for i, val in enumerate(ph_options):
        if cols[i].button(f"{val} pH", key=f"ph_{i}"):
            st.session_state.shock_values["Soil pH"] = val

    ph_custom = st.number_input(
        "Custom Soil pH Change",
        value=float(st.session_state.shock_values.get("Soil pH", 0.0)),
        step=0.1,
        key="custom_ph"
    )

    st.session_state.shock_values["Soil pH"] = ph_custom
    st.success(f"Selected Soil pH Change: {ph_custom}")

    # RUN SIMULATION
    if st.button("🚀 Run Monte Carlo Simulation (1000 runs)", key="run_sim"):
        st.session_state.screen = "simulate"
        st.rerun()
        
# ----------------------------------
# STEP 4 – SIMULATION (1000 RUNS)
# ----------------------------------

elif st.session_state.screen == "simulate":

    st.subheader("⚙ Running Monte Carlo Simulation")

    results = []

    for _ in range(1000):
        simulated = st.session_state.base_input.copy()

        simulated["temperature"] += st.session_state.shock_values["Temperature"]
        simulated["humidity"] *= (1 + st.session_state.shock_values["Humidity"] / 100)
        simulated["rainfall"] *= (1 + st.session_state.shock_values["Rainfall"] / 100)
        simulated["ph"] += st.session_state.shock_values["Soil pH"]

        pred = pipeline.predict(simulated)[0]
        results.append(pred)

    mean_yield = np.mean(results)
    simulated_revenue = mean_yield * st.session_state.area * st.session_state.price

    st.session_state.simulated_yield = mean_yield
    st.session_state.simulated_revenue = simulated_revenue

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated Yield", round(mean_yield, 2))
    col2.metric("Simulated Production",
                round(mean_yield * st.session_state.area, 2))
    col3.metric("Simulated Revenue (₹)",
                round(simulated_revenue, 2))

    if st.button("📈 View Comparison", key="compare"):
        st.session_state.screen = "comparison"
        st.rerun()

# ----------------------------------
# STEP 5 – COMPARISON
# ----------------------------------

elif st.session_state.screen == "comparison":

    st.subheader("📊 Baseline vs Scenario Comparison")

    baseline_yield = st.session_state.baseline_yield
    simulated_yield = st.session_state.simulated_yield

    baseline_revenue = st.session_state.baseline_revenue
    simulated_revenue = st.session_state.simulated_revenue

    yield_change = simulated_yield - baseline_yield
    revenue_change = simulated_revenue - baseline_revenue

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Yield", round(baseline_yield,2))
    col2.metric("Simulated Yield", round(simulated_yield,2))
    col3.metric("Yield Change", round(yield_change,2))

    st.divider()

    col4, col5, col6 = st.columns(3)

    col4.metric("Baseline Revenue (₹)", round(baseline_revenue,2))
    col5.metric("Simulated Revenue (₹)", round(simulated_revenue,2))
    col6.metric("Revenue Change (₹)", round(revenue_change,2))

    if st.button("🔄 Restart Simulation", key="restart"):
        st.session_state.screen = "context"
        st.session_state.shock_values = {
            "Temperature": 0,
            "Humidity": 0,
            "Rainfall": 0,
            "Soil pH": 0
        }
        st.rerun()



