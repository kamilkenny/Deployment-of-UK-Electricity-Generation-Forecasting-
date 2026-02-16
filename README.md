# Deployment-of-UK-Electricity-Generation-Forecasting-
This model deploy a UK electricity generation app to understanding generation dynamics in the future.

# ⚡ UK Electricity Generation Forecaster
### A.I. Predictive Modelling for National Grid Decarbonization Strategies

![Project Banner](https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?q=80&w=1000)

## 👤 Modeller Information
* **Name:** Kamil Kehinde
* **Role:** Lead Energy Modeller / Data Scientist
* **Case Study:** Great Britain’s System Operator (NESO) Generation Outputs

---

## 📖 Project Overview
The UK electricity system is transitioning rapidly. Accurate short-term forecasting is vital for balancing the grid as we integrate more intermittent renewable sources. 

This project utilizes a **Univariate Time-Series approach** to predict Total Electricity Generation. By training on historical data (2009–2026) from the **NESO Data Portal**, the model identifies seasonal rhythms, daily demand ramps, and long-term trends to provide actionable grid analytics.

### Key Objectives:
* **Grid Balancing:** Predict supply requirements to maintain frequency stability.
* **Renewable Integration:** Optimize the dispatch of zero-carbon sources.
* **Operational Efficiency:** Reduce reliance on expensive, high-carbon reserve plants.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.9+
* **Model:** XGBoost Regressor (Gradient Boosting)
* **Deployment:** Streamlit / Render / Hugging Face
* **Feature Engineering:** Time-step features (Hour, Day of Week, Month, Year)

### Model Performance:
* **R² Score:** 0.94
* **Mean Absolute Error (MAE):** ~1,240 MW
* **Validation Method:** Time-series walk-forward validation

---

## 🚀 Interactive Dashboard Features
This repository powers a live web application where users can:
1.  **Select a Forecast Horizon:** Choose a custom date range for batch predictions.
2.  **View KPI Analytics:** Instantly calculate Peak Demand, Average Load, and Total GWh volume.
3.  **Visual Trends:** Interact with high-resolution line charts of forecasted generation.
4.  **Data Export:** Download prediction results as CSV for further analysis.

---

## 📂 Repository Structure
* `app.py`: Main Streamlit application script.
* `gb_gen_time_only.pkl`: The serialized XGBoost model.
* `requirements.txt`: Python dependencies for cloud deployment.
* `kamil_profile.jpg`: Modeller identification image.

---

## 💻 How to Run Locally
1. Clone the repo:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
