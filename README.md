# 🌬️ Wind Power Generation Forecasting  
> **Real-Time Wind Turbine Power Prediction using Machine Learning & Streamlit**  
🔗 **Live App:** [Wind Power Prediction Dashboard](https://wind-power-generation-forecasting-d4thutd99pd7wa6yym4wnn.streamlit.app/)

---

## 📖 Overview  

This project leverages **Machine Learning** to forecast wind turbine power generation based on **real-time weather conditions**.  
It uses a **Random Forest Regressor** trained on historical turbine data from [Kaggle’s Wind Power Forecasting Dataset](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting), and integrates a **Streamlit web app** for interactive predictions.  

The app fetches **live meteorological data** from the [Open-Meteo API](https://open-meteo.com/), applies the same preprocessing pipeline as in training, and predicts the **expected turbine output (normalized 0–1)** in real time.  

---

## 🎯 Project Objectives  

- 🧠 Build a **machine learning model** that learns how weather affects turbine output  
- 🌤 Integrate **live weather data** for real-time forecasting  
- 📊 Provide a **visual dashboard** for energy analysts to interpret predictions  
- ⚙️ Ensure reproducible preprocessing, feature scaling, and inference pipeline  

---

## 🧩 Dataset  

**Source:** [Wind Power Generation Data (Kaggle)](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting)  

| Column | Description |
|:--------|:-------------|
| `Time` | Timestamp (hourly readings) |
| `temperature_2m` | Air temperature at 2m (°F) |
| `relativehumidity_2m` | Relative humidity (%) |
| `dewpoint_2m` | Dew point (°F) |
| `windspeed_10m`, `windspeed_100m` | Wind speeds (m/s) |
| `winddirection_10m`, `winddirection_100m` | Wind directions (°) |
| `windgusts_10m` | Wind gust speed (m/s) |
| `Power` | Normalized turbine output (0–1) |

---

## ⚙️ Model Development  

### 🧹 Data Cleaning  
- Removed nulls and duplicates  
- Converted `Time` to datetime  
- Extracted `hour` and `month` features  

### 🧪 Feature Engineering  
- Transformed wind directions (0–360°) into cyclic features using sine and cosine:  
  ```python
  df['winddir_10m_sin'] = np.sin(np.radians(df['winddirection_10m']))
  df['winddir_10m_cos'] = np.cos(np.radians(df['winddirection_10m']))


  Normalized numerical features using StandardScaler

### 🤖 Model Training

- Algorithm: RandomForestRegressor

- Split: 80% Training / 20% Testing

  Metrics:

- 🧩 RMSE: 0.14 (low error)

- 📈 R²: 0.76 (model explains ~76% of output variance)
