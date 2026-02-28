import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG + LIGHT GREEN THEME
# ----------------------------------

st.set_page_config(page_title="AgriMarket Twin", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #e8f5e9;
}
h1, h2, h3 {
    color: #1b5e20;
}
.stButton>button {
    background-color: #66bb6a;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriMarket Twin – Synthetic Market Simulation Lab")
st.write("Step-by-step yield simulation under climate & soil uncertainty.")

# ----------------------------------
# LOAD MODEL + DATA
# ----------------------------------

pipeline = joblib.load("agri_yield.pkl")
data = pd.read_csv("crop.csv")

# ----------------------------------
# SESSION STATE FOR SURVEY FLOW
# ----------------------------------

variables = [
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)",
    "Temperature",
    "Humidity",
    "Soil pH",
    "Rainfall"
]

if "step" not in st.session_state:
    st.session_state.step = 0

if "responses" not in st.session_state:
    st.session_state.responses = {}

# Progress bar
st.progress((st.session_state.step + 1) / len(variables))

# ----------------------------------
# SURVEY INPUT FLOW
# ----------------------------------

current_var = variables[st.session_state.step]
st.subheader(f"Adjust {current_var}")

base_defaults = {
    "Nitrogen (N)": 90.0,
    "Phosphorus (P)": 40.0,
    "Potassium (K)": 40.0,
    "Temperature": 25.0,
    "Humidity": 70.0,
    "Soil pH": 6.5,
    "Rainfall": 200.0
}

base_value = st.number_input(
    f"Base {current_var}",
    value=base_defaults[current_var]
)

percent_change = st.slider(
    f"{current_var} Change (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=1
)

st.session_state.responses[current_var] = {
    "base": base_value,
    "percent": percent_change
}

# Navigation buttons
col1, col2 = st.columns(2)

with col1:
    if st.session_state.step > 0:
        if st.button("⬅ Back"):
            st.session_state.step -= 1

with col2:
    if st.session_state.step < len(variables) - 1:
        if st.button("Next ➡"):
            st.session_state.step += 1

# ----------------------------------
# FINAL PAGE – SIMULATION SETTINGS
# ----------------------------------

if st.session_state.step == len(variables) - 1:

    st.subheader("🌱 Additional Settings")

    crop = st.selectbox("Crop Type", sorted(data["crop"].unique()))
    fertilizer = st.selectbox("Fertilizer Type", sorted(data["fertilizer"].unique()))
    simulations = st.selectbox("Number of Simulations", [500, 1000, 2000])
    price = st.number_input("Market Price (₹ per kg)", value=20)

    ai_toggle = st.toggle("Enable AI Smart Irrigation Tool")

    if ai_toggle:
        ai_boost = st.slider("Expected Yield Boost (%)", 5, 25, 12)
        tool_cost = st.number_input("AI Tool Cost (₹ per hectare)", value=3000)
    else:
        ai_boost = 0
        tool_cost = 0

    # ----------------------------------
    # MONTE CARLO FUNCTION
    # ----------------------------------

    def monte_carlo(n_sim=1000):

        results = []

        for _ in range(n_sim):

            simulated = {}

            for var in variables:
                base = st.session_state.responses[var]["base"]
                percent = st.session_state.responses[var]["percent"] / 100
                adjusted = base * (1 + percent)
                simulated[var.split(" ")[0]] = adjusted

            df = pd.DataFrame([{
                "N": simulated["Nitrogen"],
                "P": simulated["Phosphorus"],
                "K": simulated["Potassium"],
                "temperature": simulated["Temperature"],
                "humidity": simulated["Humidity"],
                "ph": simulated["Soil"],
                "rainfall": simulated["Rainfall"],
                "crop": crop,
                "fertilizer": fertilizer
            }])

            # Add Noise
            df["rainfall"] = np.random.normal(df["rainfall"], df["rainfall"] * 0.10)
            df["temperature"] = np.random.normal(df["temperature"], 1.5)
            df["humidity"] = np.random.normal(df["humidity"], df["humidity"] * 0.05)

            pred = pipeline.predict(df)[0]
            results.append(pred)

        return np.array(results)

    # ----------------------------------
    # RUN SIMULATION
    # ----------------------------------

    if st.button("🚀 Run Simulation"):

        results = monte_carlo(simulations)

        mean_yield = np.mean(results)
        worst = np.percentile(results, 5)
        best = np.percentile(results, 95)

        baseline_revenue = mean_yield * price

        st.subheader("📈 Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Yield", round(mean_yield, 2))
        col2.metric("Worst Case (5%)", round(worst, 2))
        col3.metric("Best Case (95%)", round(best, 2))

        st.metric("Expected Revenue (₹)", round(baseline_revenue, 2))

        if ai_toggle:
            intervention = results * (1 + ai_boost / 100)
            mean_ai = np.mean(intervention)
            revenue_ai = mean_ai * price
            roi = (revenue_ai - baseline_revenue) / tool_cost

            st.subheader("🤖 AI Tool Impact")

            col4, col5 = st.columns(2)
            col4.metric("AI Expected Yield", round(mean_ai, 2))
            col5.metric("AI Revenue (₹)", round(revenue_ai, 2))
            st.metric("Estimated ROI", round(roi, 2))

        st.subheader("📊 Yield Distribution")

        fig, ax = plt.subplots()
        ax.hist(results, bins=30, alpha=0.6, label="Baseline")

        if ai_toggle:
            ax.hist(intervention, bins=30, alpha=0.6, label="AI Tool")

        ax.set_xlabel("Predicted Yield")
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)
