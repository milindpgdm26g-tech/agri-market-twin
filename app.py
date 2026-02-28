import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(page_title="AgriMarket Twin", layout="wide")

st.title("🌾 AgriMarket Twin – Smart Yield Simulation Engine")
st.write("Run Monte Carlo simulations to estimate crop yield under climate and soil uncertainty.")

# ----------------------------------
# LOAD MODEL + DATA
# ----------------------------------

pipeline = joblib.load("agri_yield.pkl")
data = pd.read_csv("crop.csv")

# ----------------------------------
# BASE INPUT SECTION
# ----------------------------------

st.subheader("📍 Base Agricultural Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", value=90.0)
    P = st.number_input("Phosphorus (P)", value=40.0)
    K = st.number_input("Potassium (K)", value=40.0)

with col2:
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=70.0)
    ph = st.number_input("Soil pH", value=6.5)

with col3:
    rainfall = st.number_input("Rainfall (mm)", value=200.0)
    crop = st.selectbox("Crop Type", sorted(data["crop"].unique()))
    fertilizer = st.selectbox("Fertilizer Type", sorted(data["fertilizer"].unique()))

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

# ----------------------------------
# STRUCTURED SCENARIO LEVELS
# ----------------------------------

st.subheader("📊 Climate Stress Levels")

level_map_percent = {1:5, 2:10, 3:15, 4:20, 5:30}
level_map_temp = {1:1, 2:2, 3:3, 4:4, 5:5}

rainfall_dir = st.selectbox("Rainfall Direction", ["No Change","Increase","Decrease"])
rainfall_lvl = st.selectbox("Rainfall Level (1-5)", [1,2,3,4,5])

temp_dir = st.selectbox("Temperature Direction", ["No Change","Increase","Decrease"])
temp_lvl = st.selectbox("Temperature Level (1-5)", [1,2,3,4,5])

humidity_dir = st.selectbox("Humidity Direction", ["No Change","Increase","Decrease"])
humidity_lvl = st.selectbox("Humidity Level (1-5)", [1,2,3,4,5])

n_dir = st.selectbox("Nitrogen Direction", ["No Change","Increase","Decrease"])
n_lvl = st.selectbox("Nitrogen Level (1-5)", [1,2,3,4,5])

# ----------------------------------
# AI TOOL TOGGLE
# ----------------------------------

st.subheader("🤖 AI Tool Intervention")

ai_toggle = st.toggle("Enable AI Smart Irrigation Tool")

if ai_toggle:
    ai_boost = st.slider("Expected Yield Boost (%)", 5, 25, 12)
    tool_cost = st.number_input("AI Tool Cost (₹ per hectare)", value=3000)
else:
    ai_boost = 0
    tool_cost = 0

# ----------------------------------
# MONTE CARLO SIMULATION
# ----------------------------------

def monte_carlo(pipeline, base_input, n_sim=1000):

    results = []

    for _ in range(n_sim):

        simulated = base_input.copy()

        # Structured Scenario Adjustments

        if rainfall_dir != "No Change":
            percent = level_map_percent[rainfall_lvl] / 100
            if rainfall_dir == "Increase":
                simulated["rainfall"] *= (1 + percent)
            else:
                simulated["rainfall"] *= (1 - percent)

        if temp_dir != "No Change":
            change = level_map_temp[temp_lvl]
            if temp_dir == "Increase":
                simulated["temperature"] += change
            else:
                simulated["temperature"] -= change

        if humidity_dir != "No Change":
            percent = level_map_percent[humidity_lvl] / 100
            if humidity_dir == "Increase":
                simulated["humidity"] *= (1 + percent)
            else:
                simulated["humidity"] *= (1 - percent)

        if n_dir != "No Change":
            percent = level_map_percent[n_lvl] / 100
            if n_dir == "Increase":
                simulated["N"] *= (1 + percent)
            else:
                simulated["N"] *= (1 - percent)

        # Add Monte Carlo Noise
        simulated["rainfall"] = np.random.normal(simulated["rainfall"], simulated["rainfall"] * 0.10)
        simulated["temperature"] = np.random.normal(simulated["temperature"], 1.5)
        simulated["humidity"] = np.random.normal(simulated["humidity"], simulated["humidity"] * 0.05)

        pred = pipeline.predict(simulated)[0]
        results.append(pred)

    return np.array(results)

# ----------------------------------
# RUN SIMULATION
# ----------------------------------

st.subheader("⚙ Simulation Settings")

simulations = st.selectbox("Number of Simulations", [500,1000,2000])
price = st.number_input("Market Price (₹ per kg)", value=20)

if st.button("🚀 Run Simulation"):

    results = monte_carlo(pipeline, base_input, simulations)

    mean_yield = np.mean(results)
    worst = np.percentile(results,5)
    best = np.percentile(results,95)

    baseline_revenue = mean_yield * price

    st.subheader("📈 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Yield", round(mean_yield,2))
    col2.metric("Worst Case (5%)", round(worst,2))
    col3.metric("Best Case (95%)", round(best,2))

    st.metric("Expected Revenue (₹)", round(baseline_revenue,2))

    # AI Comparison
    if ai_toggle:
        intervention = results * (1 + ai_boost/100)
        mean_ai = np.mean(intervention)
        revenue_ai = mean_ai * price
        roi = (revenue_ai - baseline_revenue) / tool_cost

        st.subheader("🤖 AI Tool Impact")

        col4, col5 = st.columns(2)
        col4.metric("AI Expected Yield", round(mean_ai,2))
        col5.metric("AI Revenue (₹)", round(revenue_ai,2))

        st.metric("Estimated ROI", round(roi,2))

    # Plot
    st.subheader("📊 Yield Distribution")

    fig, ax = plt.subplots()
    ax.hist(results, bins=30, alpha=0.6, label="Baseline")

    if ai_toggle:
        ax.hist(intervention, bins=30, alpha=0.6, label="AI Tool")

    ax.set_xlabel("Predicted Yield")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)