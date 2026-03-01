import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    font-size: 30px !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    background-color: #2f2f2f !important;
    border: 2px solid #2e7d32 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] span { color: white !important; }

.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    border-radius: 10px !important;
    height: 3em !important;
    font-weight: 600 !important;
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
# SAFE SESSION STATE INIT
# ----------------------------------

defaults = {
    "screen": "context",
    "shock_values": {
        "Temperature": 0,
        "Humidity": 0,
        "Rainfall": 0,
        "Soil pH": 0
    },
    "volatility": 0,
    "prob_loss": 0,
    "best_case": 0,
    "worst_case": 0,
    "best_revenue": 0,
    "worst_revenue": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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

    if st.button("📊 Predict Baseline Yield"):

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

    if st.button("🚀 Proceed to Scenario Simulation"):
        st.session_state.screen = "tiles"
        st.rerun()

# ----------------------------------
# STEP 3 – SCENARIO SELECTION
# ----------------------------------

elif st.session_state.screen == "tiles":

    st.subheader("🧩 Scenario Selection")

    st.session_state.shock_values["Temperature"] = st.number_input(
        "Temperature Change (°C)", value=0, step=1)

    st.session_state.shock_values["Humidity"] = st.number_input(
        "Humidity Change (%)", value=0, step=1)

    st.session_state.shock_values["Rainfall"] = st.number_input(
        "Rainfall Change (%)", value=0, step=1)

    st.session_state.shock_values["Soil pH"] = st.number_input(
        "Soil pH Change", value=0.0, step=0.1)

    if st.button("🚀 Run Monte Carlo Simulation (1000 runs)"):
        st.session_state.screen = "simulate"
        st.rerun()

# ----------------------------------
# STEP 4 – PROBABILISTIC MONTE CARLO
# ----------------------------------

elif st.session_state.screen == "simulate":

    st.subheader("⚙ Running Probabilistic Monte Carlo Simulation")

    results = []
    baseline_yield = st.session_state.baseline_yield

    for _ in range(1000):
        simulated = st.session_state.base_input.copy()

        simulated["temperature"] += st.session_state.shock_values["Temperature"]
        simulated["humidity"] *= (1 + st.session_state.shock_values["Humidity"] / 100)
        simulated["rainfall"] *= (1 + st.session_state.shock_values["Rainfall"] / 100)
        simulated["ph"] += st.session_state.shock_values["Soil pH"]

        simulated["temperature"] = np.random.normal(simulated["temperature"], 1.5)
        simulated["rainfall"] = np.random.normal(simulated["rainfall"], simulated["rainfall"] * 0.10)
        simulated["humidity"] = np.random.normal(simulated["humidity"], simulated["humidity"] * 0.05)
        simulated["ph"] = np.random.normal(simulated["ph"], 0.2)

        results.append(pipeline.predict(simulated)[0])

    results = np.array(results)

    mean_yield = np.mean(results)
    std_yield = np.std(results)
    worst_case = np.percentile(results, 5)
    best_case = np.percentile(results, 95)
    prob_loss = np.mean(results < baseline_yield) * 100

    mean_revenue = mean_yield * st.session_state.area * st.session_state.price

    st.session_state.simulated_yield = mean_yield
    st.session_state.simulated_revenue = mean_revenue
    st.session_state.best_case = best_case
    st.session_state.worst_case = worst_case
    st.session_state.volatility = std_yield
    st.session_state.prob_loss = prob_loss

    col1, col2, col3 = st.columns(3)
    col1.metric("Worst Case (5%)", round(worst_case, 2))
    col2.metric("Expected Yield", round(mean_yield, 2))
    col3.metric("Best Case (95%)", round(best_case, 2))

    st.metric("Volatility (Std Dev)", round(std_yield, 2))
    st.metric("Probability of Loss", f"{round(prob_loss, 2)}%")

    if st.button("📈 View Comparison"):
        st.session_state.screen = "comparison"
        st.rerun()

# ----------------------------------
# STEP 5 – COMPARISON
# ----------------------------------

elif st.session_state.screen == "comparison":

    st.subheader("📊 Baseline vs Scenario Comparison")

    baseline_yield = st.session_state.baseline_yield
    mean_yield = st.session_state.simulated_yield

    baseline_revenue = st.session_state.baseline_revenue
    mean_revenue = st.session_state.simulated_revenue

    yield_change = mean_yield - baseline_yield
    yield_percent = (yield_change / baseline_yield) * 100

    revenue_change = mean_revenue - baseline_revenue
    revenue_percent = (revenue_change / baseline_revenue) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Yield", round(baseline_yield, 2))
    col2.metric("Expected Yield", round(mean_yield, 2))
    col3.metric("Yield Change", round(yield_change, 2),
                f"{round(yield_percent, 2)}%")

    st.divider()

    col4, col5, col6 = st.columns(3)
    col4.metric("Baseline Revenue", round(baseline_revenue, 2))
    col5.metric("Expected Revenue", round(mean_revenue, 2))
    col6.metric("Revenue Change", round(revenue_change, 2),
                f"{round(revenue_percent, 2)}%")

    st.divider()

    st.metric("Probability of Loss", f"{round(st.session_state.prob_loss, 2)}%")
    st.metric("Yield Volatility", round(st.session_state.volatility, 2))

    if st.button("🔄 Restart Simulation"):
        st.session_state.screen = "context"
        st.rerun()
