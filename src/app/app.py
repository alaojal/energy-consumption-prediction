import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import streamlit as st
import pandas as pd

#from src.inference_pipeline.inference import predict
from src.inference_pipeline.inference import predict

# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="Energy Consumption Portal",
    page_icon="💡",
    layout="wide"
)

PREDICTION_COLUMN = "predicted_AEP_MW"
MODEL_NAME = "GradientBoostingRegressor"
MODEL_VERSION = "1.0.0"

# =========================================
# Header section
# ========================================
st.markdown("""
# 💡 Energy Consumption Portal

### AI-Driven Energy Consumption Prediction

This analytics portal leverages machine learning to predict hourly energy consumption from engineered date features.
""")


# ==========================================
# Energy Consumption Prediction
# ==========================================
st.header("💡Energy Consumption Prediction")
st.markdown("""
**Objective**
Predict hourly energy consumption using the following engineered date features:

- Year
- Month
- Day
- Hour
""")

# Add sidebar
with st.sidebar:
    st.header("Model Information")

    st.markdown("**Model**")
    st.write(MODEL_NAME)

    st.markdown("**Version**")
    st.write(MODEL_VERSION)


with st.form("energy_prediction_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year = st.number_input(
            "Year",
            min_value=1900,
            max_value=2100,
            value=2020,
            step=1,
            help="Calendar year."            
        )

    with col2:
        month = st.selectbox(
            "Month", 
            options=range(1, 13),
            index=0, 
            help="Month of the year (1–12)."            
        )
    with col3:
        day = st.number_input(
             "Day", 
            min_value=1,
            max_value=31,
            value=1,
            step=1,
            help="Day of the month (1–31)."
        ) 

    with col4:
            hour = st.selectbox(
                "Hour",
                options=list(range(24)), 
                index=12, 
                format_func=lambda x: f"{x:02d}",
                help="Hour of the day (0–23)."              
            )

    submit_energy = st.form_submit_button("🪫 Predict Energy Consumption")   

    if submit_energy:
         df = pd.DataFrame(
              {
              "year": [year],
              "month": [month],
              "day": [day],
              "hour": [hour]
         } 
         )

         try:
              with st.spinner("Generating energy consumption prediction..."):
                   result = predict(df)

                   prediction = result[PREDICTION_COLUMN].iloc[0]

                   st.success("✅ Prediction completed successfully")

                   st.metric(
                        label="📊 Predicted Hourly Energy Consumption",
                        value=f"{prediction:,.2f} MW"
                   )

                   st.divider() 

                   st.subheader("Prediction Inputs")

                   st.dataframe(
                         df,
                         hide_index=True,
                         use_container_width=True
                   )

                   st.info(
                         "Prediction generated using the trained GradientBoostingRegressor model."
                   )
         except Exception as exc:
              st.error(f"Prediction failed: {exc}")

# Add footer
st.divider()

st.caption(
    "Powered by Streamlit, FastAPI, and GradientBoostingRegressor."
)