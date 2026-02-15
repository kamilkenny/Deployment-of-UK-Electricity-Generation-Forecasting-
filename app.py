import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from PIL import Image

# 1. Page Config - Wide mode with a custom theme feel
st.set_page_config(
    page_title="UK National Grid Forecast | Kamil Kehinde",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CENTERING & STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Centering logic for images */
    .centered-image {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar - Sophisticated Branding
with st.sidebar:
    # Centering the Profile Picture
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    try:
        profile_img = Image.open("kamil_profile.jpeg")
        st.image(profile_img, width=180)
    except:
        st.write("👤")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: white;'>Kamil Kehinde</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b0b0b0;'>Lead Energy Modeller</p>", unsafe_allow_html=True)
    
    st.divider()
    st.write("🛠️ **Model Architecture**")
    st.caption("XGBoost Regressor (Univariate)")
    st.caption("Data Source: NESO Historicals")
    st.caption("Training Horizon: 2009-2026")

# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load('gb_gen_time_only.pkl')

model = load_model()

# 4. Header Section
col_title, col_img = st.columns([2, 1])
with col_title:
    st.title("⚡ UK Electricity Generation Dashboard")
    st.info("High-precision predictive analytics for National Grid decarbonization strategies.")
with col_img:
    # Centered illustrative image in the main body
    st.image("https://images.unsplash.com/photo-1466611653911-95081537e5b7?q=80&w=400", use_container_width=True)

st.divider()

# 5. Advanced Range Selection
st.subheader("📅 Forecast Horizon Configuration")
c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    selected_range = st.date_input(
        "Observation Window",
        value=(datetime.date(2026, 2, 15), datetime.date(2026, 2, 18)),
        help="Select a range to generate a batch forecast."
    )
with c2:
    agg_type = st.selectbox("Aggregation Mode", ["Hourly (30m)", "Daily Average", "Weekly Sum"])
with c3:
    st.write("") # Spacer
    st.write("") # Spacer
    predict_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# 6. Analytics Logic
if predict_btn and len(selected_range) == 2:
    start_date, end_date = selected_range
    
    # Generate Time Index
    predict_index = pd.date_range(
        start=datetime.datetime.combine(start_date, datetime.time(0, 0)),
        end=datetime.datetime.combine(end_date, datetime.time(23, 30)),
        freq='30T'
    )
    
    # Features
    predict_df = pd.DataFrame(index=predict_index)
    predict_df['hour'] = predict_df.index.hour
    predict_df['dayofweek'] = predict_df.index.dayofweek
    predict_df['month'] = predict_df.index.month
    predict_df['year'] = predict_df.index.year
    
    # Predict
    predict_df['Forecast_MW'] = model.predict(predict_df)
    
    # 7. Sophisticated Metrics Section
    st.markdown("### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    avg_gen = predict_df['Forecast_MW'].mean()
    peak_gen = predict_df['Forecast_MW'].max()
    total_energy = (predict_df['Forecast_MW'].sum() * 0.5) / 1000 # Convert MWh to GWh
    low_load = predict_df['Forecast_MW'].min()

    kpi1.metric("Avg Generation", f"{avg_gen:,.0f} MW", "Stable")
    kpi2.metric("Peak Demand", f"{peak_gen:,.0f} MW", delta=f"{peak_gen-avg_gen:,.0f} Variance", delta_color="inverse")
    kpi3.metric("Total Volume", f"{total_energy:,.2f} GWh")
    kpi4.metric("Baseload Min", f"{low_load:,.0f} MW")

    # 8. Interactive Plot
    st.divider()
    st.subheader(f"📊 Forecasted Generation Trend")
    
    # Use tabs for different views
    tab1, tab2 = st.tabs(["Visual Chart", "Raw Data Table"])
    
    with tab1:
        st.line_chart(predict_df['Forecast_MW'], color="#1f77b4")
        [Image of interactive time series electricity demand chart]
    
    with tab2:
        st.dataframe(predict_df[['Forecast_MW']], use_container_width=True)

    # 9. Footer & Export
    st.divider()
    col_dl, col_foot = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "📥 Export Analysis (CSV)",
            data=predict_df.to_csv(),
            file_name=f"UK_Grid_Forecast_{start_date}.csv",
            mime="text/csv"
        )
    with col_foot:
        st.caption("Developed by Kamil Kehinde | Version 2.1 | © 2026 Energy Analytics Inc.")

elif predict_btn:
    st.error("Please select a valid date range (Start and End).")
