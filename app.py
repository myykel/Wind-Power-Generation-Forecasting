# ==========================================================
# 🌬️ Wind Power Prediction App
# Author: Michael Eboji
# Purpose: Predict turbine power output from weather data
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Wind Power Predictor",
    page_icon="💨",
    layout="centered"
)

st.title("🌬️ Wind Power Prediction App")
st.markdown("""
This app uses a **Machine Learning model** trained on historical wind turbine data  
to predict how much power (normalized 0–1 scale) a turbine will produce based on today's weather.
""")

# ----------------------------------------------------------
# 2. Load Model and Scaler
# ----------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    rf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return rf, scaler

rf, scaler = load_model_and_scaler()

# ----------------------------------------------------------
# 3. Sidebar Inputs
# ----------------------------------------------------------
st.sidebar.header("🌤 Enter Today's Weather Conditions")

temperature_2m = st.sidebar.number_input("Temperature (°F)", value=60.0, step=0.5)
relativehumidity_2m = st.sidebar.number_input("Relative Humidity (%)", value=70.0, step=1.0)
dewpoint_2m = st.sidebar.number_input("Dew Point (°F)", value=52.0, step=0.5)
windspeed_10m = st.sidebar.number_input("Wind Speed at 10m (m/s)", value=8.5, step=0.1)
windspeed_100m = st.sidebar.number_input("Wind Speed at 100m (m/s)", value=10.2, step=0.1)
winddirection_10m = st.sidebar.number_input("Wind Direction at 10m (°)", value=240, step=1)
winddirection_100m = st.sidebar.number_input("Wind Direction at 100m (°)", value=245, step=1)
windgusts_10m = st.sidebar.number_input("Wind Gusts at 10m (m/s)", value=12.1, step=0.1)

# ----------------------------------------------------------
# 4. Predict Button
# ----------------------------------------------------------
if st.button("🔮 Predict Power Output"):

    # Create DataFrame from inputs
    new_data = pd.DataFrame({
        'temperature_2m': [temperature_2m],
        'relativehumidity_2m': [relativehumidity_2m],
        'dewpoint_2m': [dewpoint_2m],
        'windspeed_10m': [windspeed_10m],
        'windspeed_100m': [windspeed_100m],
        'winddirection_10m': [winddirection_10m],
        'winddirection_100m': [winddirection_100m],
        'windgusts_10m': [windgusts_10m]
    })

    # ----------------------------------------------------------
    # 5. Apply same preprocessing as training
    # ----------------------------------------------------------
    new_data['winddir_10m_rad'] = np.radians(new_data['winddirection_10m'])
    new_data['winddir_100m_rad'] = np.radians(new_data['winddirection_100m'])

    new_data['winddir_10m_sin'] = np.sin(new_data['winddir_10m_rad'])
    new_data['winddir_10m_cos'] = np.cos(new_data['winddir_10m_rad'])
    new_data['winddir_100m_sin'] = np.sin(new_data['winddir_100m_rad'])
    new_data['winddir_100m_cos'] = np.cos(new_data['winddir_100m_rad'])

    new_data.drop(columns=['winddirection_10m', 'winddirection_100m',
                           'winddir_10m_rad', 'winddir_100m_rad'], inplace=True)

    trained_features = scaler.feature_names_in_

    for col in trained_features:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[trained_features]

    # ----------------------------------------------------------
    # 6. Scale and Predict
    # ----------------------------------------------------------
    new_scaled = scaler.transform(new_data)
    predicted_power = rf.predict(new_scaled)[0]

    # ----------------------------------------------------------
    # 7. Display Result
    # ----------------------------------------------------------
    st.success(f"⚡ Predicted Power Output: **{predicted_power:.2f}** (normalized 0–1 scale)")
    st.caption("1.0 = Maximum turbine capacity, 0.0 = No power")

    # Visual gauge
    st.progress(predicted_power)
    st.write(f"**Performance:** {predicted_power*100:.1f}% of maximum output")

    # Simple bar chart
    fig, ax = plt.subplots()
    ax.bar(["Predicted Power"], [predicted_power], color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Power Output (0–1)")
    ax.set_title("Predicted Turbine Performance")
    st.pyplot(fig)

    # ----------------------------------------------------------
    # 8. Optional: Download CSV
    # ----------------------------------------------------------
    new_data['Predicted_Power'] = predicted_power
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction", csv, "predicted_output.csv", "text/csv")

# ----------------------------------------------------------
# 9. Footer
# ----------------------------------------------------------
st.markdown("""
---
Created with ❤️ by **Michael Eboji**  
[GitHub Repository](https://github.com/myykel) | [Streamlit Cloud](https://streamlit.io)
""")
