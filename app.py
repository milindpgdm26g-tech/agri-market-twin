import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG + CLEAN GREEN UI
# ----------------------------------

st.set_page_config(page_title="AgriMarket Twin", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #e8f5e9; }

h1 { color: black !important; font-weight: 700; }
h2, h3 { color: #1b5e20 !important; }

div, label, p, span { color: black !important; }

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

st.title("🌾 AgriMarket Twin – Synthetic Market Simulation Lab")

# ----------------------------------
# LOAD MODEL + DATA
# ----------------------------------

pipeline = joblib.load("agri_yield.pkl")
data = pd.read_csv("crop.csv")

# ----------------------------------
# SESSION STATE CONTROL
# ----------------------------------

if "screen" not in st.session_state:
    st.session_state.screen = "context"

if "step" not in st.session_state:
    st.session_state.step = 0

if "scenario_choices" not in st.session_state:
    st.session_state.scenario_choices = {}

# ----------------------------------
# SCENARIO STRUCTURE (5 options each)
# ----------------------------------

scenario_structure = {
    "Rainfall": [-30, -15, 0, 15, 30],       # %
    "Temperature": [-3, -1, 0, 1, 3],        # °C
    "Humidity": [-20, -10, 0, 10, 20],       # %
    "Nitrogen": [-25, -10, 0, 10, 25]        # %
}

variables = list(scenario_structure.keys())

# ----------------------------------
# SCREEN 1 – MARKET CONTEXT
# ----------------------------------

if st.session_state.screen == "context":

    st.subheader("📍 Market Context Setup")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("Select State", sorted(data["State"].unique()))
        crop = st.selectbox("Select Crop", sorted(data["Crop"].unique()))

    with col2:
        district = st.selectbox("Select District", sorted(data["District"].unique()))
        fertilizer = st.selectbox("Fertilizer Type", sorted(data["Fertilizer"].unique()))

    price = st.number_input("Market Price (₹ per kg)", value=20)

    if st.button("🚀 Start Simulation"):
        st.session_state.state = state
        st.session_state.district = district
        st.session_state.crop = crop
        st.session_state.fertilizer = fertilizer
        st.session_state.price = price

        # Base defaults (can be dynamic later)
        st.session_state.base_input = pd.DataFrame([{
            "N": 90.0,
            "P": 40.0,
            "K": 40.0,
            "temperature": 25.0,
            "humidity": 70.0,
            "ph": 6.5,
            "rainfall": 200.0,
            "crop": crop,
            "fertilizer": fertilizer
        }])

        st.session_state.screen = "tiles"
        st.rerun()

# ----------------------------------
# SCREEN 2 – TILE SELECTION FLOW
# ----------------------------------

elif st.session_state.screen == "tiles":

    st.progress((st.session_state.step + 1) / len(variables))

    current_var = variables[st.session_state.step]
    st.subheader(f"🧩 Select {current_var} Scenario")

    options = scenario_structure[current_var]
    cols = st.columns(5)

    for i, value in enumerate(options):

        label = f"{value}%" if current_var != "Temperature" else f"{value}°C"

        if cols[i].button(label, key=f"{current_var}_{i}"):

            st.session_state.scenario_choices[current_var] = value

            if st.session_state.step < len(variables) - 1:
                st.session_state.step += 1
            else:
                st.session_state.screen = "simulate"

            st.rerun()

# ----------------------------------
# SCREEN 3 – RUN SIMULATION
# ----------------------------------

elif st.session_state.screen == "simulate":

    st.subheader("⚙ Simulation Settings")

    simulations = st.selectbox("Number of Simulations", [500, 1000, 2000])

    def monte_carlo_with_tiles(n_sim=1000):

        results = []

        for _ in range(n_sim):

            simulated = st.session_state.base_input.copy()

            for var, change in st.session_state.scenario_choices.items():

                if var == "Rainfall":
                    simulated["rainfall"] *= (1 + change/100)

                elif var == "Temperature":
                    simulated["temperature"] += change

                elif var == "Humidity":
                    simulated["humidity"] *= (1 + change/100)

                elif var == "Nitrogen":
                    simulated["N"] *= (1 + change/100)

            # Add uncertainty
            simulated["rainfall"] = np.random.normal(simulated["rainfall"], simulated["rainfall"] * 0.10)
            simulated["temperature"] = np.random.normal(simulated["temperature"], 1.5)
            simulated["humidity"] = np.random.normal(simulated["humidity"], simulated["humidity"] * 0.05)

            pred = pipeline.predict(simulated)[0]
            results.append(pred)

        return np.array(results)

    if st.button("🚀 Run Simulation"):

        results = monte_carlo_with_tiles(simulations)

        mean_yield = np.mean(results)
        worst = np.percentile(results, 5)
        best = np.percentile(results, 95)

        revenue = mean_yield * st.session_state.price

        st.subheader("📈 Simulation Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Yield", round(mean_yield, 2))
        col2.metric("Worst Case (5%)", round(worst, 2))
        col3.metric("Best Case (95%)", round(best, 2))

        st.metric("Expected Revenue (₹)", round(revenue, 2))

        st.subheader("📊 Yield Distribution")

        fig, ax = plt.subplots()
        ax.hist(results, bins=30, alpha=0.7)
        ax.set_xlabel("Predicted Yield")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    if st.button("🔄 Reset Simulation"):
        st.session_state.screen = "context"
        st.session_state.step = 0
        st.session_state.scenario_choices = {}
        st.rerun()

