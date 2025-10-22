import pandas as pd
import numpy as np
import joblib

# Load model and scaler
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load new data
new_data = pd.read_csv("new_conditions.csv")

# -----------------------------
# 1️⃣ Apply the same preprocessing as training
# -----------------------------

# Convert directions to sin/cos (since model used these)
new_data['winddir_10m_rad'] = np.radians(new_data['winddirection_10m'])
new_data['winddir_100m_rad'] = np.radians(new_data['winddirection_100m'])

new_data['winddir_10m_sin'] = np.sin(new_data['winddir_10m_rad'])
new_data['winddir_10m_cos'] = np.cos(new_data['winddir_10m_rad'])
new_data['winddir_100m_sin'] = np.sin(new_data['winddir_100m_rad'])
new_data['winddir_100m_cos'] = np.cos(new_data['winddir_100m_rad'])

# Drop unnecessary columns
new_data = new_data.drop(columns=['winddirection_10m', 'winddirection_100m',
                                  'winddir_10m_rad', 'winddir_100m_rad'])

# 2️⃣ Ensure feature alignment
# -----------------------------
# Get column names the model was trained on
trained_features = scaler.feature_names_in_

# Add any missing columns (set them to 0)
for col in trained_features:
    if col not in new_data.columns:
        new_data[col] = 0

# Drop extra columns (keep only what model expects)
new_data = new_data[trained_features]

# -----------------------------
# 3️⃣ Scale & Predict
# -----------------------------
new_scaled = scaler.transform(new_data)
predicted_power = rf.predict(new_scaled)

# -----------------------------
# 4️⃣ Save and display
# -----------------------------
new_data['Predicted_Power'] = predicted_power
new_data.to_csv("predicted_output.csv", index=False)
print("✅ Prediction complete! Check predicted_output.csv")
print(new_data[['Predicted_Power']])
