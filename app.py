import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #e8f5e9;
}

/* Headings */
h1 { color: black !important; font-weight: 700; }
h2, h3 { color: #1b5e20 !important; }

/* ----------------------------- */
/* SELECTBOX COMPLETE OVERRIDE  */
/* ----------------------------- */

/* Selected box */
div[data-baseweb="select"] > div {
    background-color: transparent !important;
    border: 1px solid #2e7d32 !important;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown menu container */
div[role="listbox"] {
    background-color: #1f1f1f !important;
}

/* Dropdown option text */
div[role="option"] {
    color: white !important;
    background-color: #1f1f1f !important;
}

/* Hover state */
div[role="option"]:hover {
    background-color: #2e7d32 !important;
    color: white !important;
}

/* Selected option highlight */
div[aria-selected="true"] {
    background-color: #2e7d32 !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1b5e20;
}

</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriMarket Twin – Multi Scenario Simulation")

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
# STEP 1 – FARMER INFO
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

    # Auto-fill
    soil_filtered = soil_data[(soil_data["State"] == state) &
                              (soil_data["Region"] == region)]
    weather_filtered = weather_data[(weather_data["State"] == state) &
                                    (weather_data["Region"] == region)]

    if not soil_filtered.empty:
        N = int(soil_filtered["N"].mean())
        P = int(soil_filtered["P"].mean())
        K = int(soil_filtered["K"].mean())
        ph = float(round(soil_filtered["pH"].mean(), 1))   # FIXED
    else:
        N, P, K, ph = 90, 40, 40, 6.5

    if not weather_filtered.empty:
        temperature = int(weather_filtered["Temperature"].mean())   # FIXED
        humidity = int(weather_filtered["Humidity"].mean())         # FIXED
        rainfall = int(weather_filtered["Rainfall"].mean())         # FIXED
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

    if st.button("🚀 Proceed to Scenario Simulation"):

        st.session_state.base_input = pd.DataFrame([{
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

        st.session_state.area = area
        st.session_state.price = price
        st.session_state.screen = "tiles"
        st.rerun()

# ----------------------------------
# STEP 2 – MULTI TILE FLOW
# ----------------------------------

elif st.session_state.screen == "tiles":

    variables = ["Temperature", "Humidity", "Rainfall", "Soil pH"]

    scenario_map = {
        "Temperature": [-3, -1, 0, 1, 3],        # °C
        "Humidity": [-20, -10, 0, 10, 20],       # %
        "Rainfall": [-30, -15, 0, 15, 30],       # %
        "Soil pH": [-1, -0.5, 0, 0.5, 1]         # pH units
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

        if unit == "%":
            label = f"{value}%"
        elif unit == "°C":
            label = f"{value} °C"
        else:
            label = f"{value} pH"

        if cols[i].button(label):
            st.session_state.shock_values[current_var] = value

    if current_var in st.session_state.shock_values:
        selected = st.session_state.shock_values[current_var]
        st.info(f"Selected Change: {selected} {unit_map[current_var]}")

    if st.button("Next ➡"):

        if current_var not in st.session_state.shock_values:
            st.warning("Please select a scenario before proceeding.")
        else:
            if st.session_state.step < len(variables) - 1:
                st.session_state.step += 1
            else:
                st.session_state.screen = "simulate"
            st.rerun()

# ----------------------------------
# STEP 3 – SIMULATION
# ----------------------------------

elif st.session_state.screen == "simulate":

    st.subheader("⚙ Running Monte Carlo Simulation")

    st.markdown("### 📋 Selected Scenario Changes")
    st.write(f"Temperature: {st.session_state.shock_values['Temperature']} °C")
    st.write(f"Humidity: {st.session_state.shock_values['Humidity']} %")
    st.write(f"Rainfall: {st.session_state.shock_values['Rainfall']} %")
    st.write(f"Soil pH: {st.session_state.shock_values['Soil pH']} pH units")

    simulations = 1000
    results = []

    for _ in range(simulations):

        simulated = st.session_state.base_input.copy()

        simulated["temperature"] += st.session_state.shock_values["Temperature"]
        simulated["humidity"] *= (1 + st.session_state.shock_values["Humidity"]/100)
        simulated["rainfall"] *= (1 + st.session_state.shock_values["Rainfall"]/100)
        simulated["ph"] += st.session_state.shock_values["Soil pH"]

        simulated["rainfall"] = np.random.normal(simulated["rainfall"], simulated["rainfall"] * 0.10)
        simulated["temperature"] = np.random.normal(simulated["temperature"], 1.5)
        simulated["humidity"] = np.random.normal(simulated["humidity"], simulated["humidity"] * 0.05)

        pred = pipeline.predict(simulated)[0]
        results.append(pred)

    results = np.array(results)

    mean_yield = np.mean(results)
    total_production = mean_yield * st.session_state.area
    total_revenue = total_production * st.session_state.price

    col1, col2, col3 = st.columns(3)
    col1.metric("Yield (per hectare)", round(mean_yield, 2))
    col2.metric("Total Production", round(total_production, 2))
    col3.metric("Revenue (₹)", round(total_revenue, 2))

    fig, ax = plt.subplots()
    ax.hist(results, bins=30)
    ax.set_xlabel("Predicted Yield")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    if st.button("🔄 Restart"):
        st.session_state.screen = "context"
        st.session_state.step = 0
        st.session_state.shock_values = {}
        st.rerun()



