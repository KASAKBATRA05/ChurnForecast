# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and required artifacts
pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>📊 Customer Churn Prediction App</h1>",
    unsafe_allow_html=True
)
st.write("Use this form to check if a customer is likely to churn.")

# ✅ Binary numeric columns like SeniorCitizen (must match exact column name)
binary_numeric_features = ["SeniorCitizen"]

# Two-column layout
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        if feature in encoders:
            options = list(encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", options)
            encoded = encoders[feature].transform([selected])[0]
            input_data[feature] = encoded

        elif feature in binary_numeric_features:
            # Final fix: force only Yes/No option
            selected = st.selectbox(f"{feature}", ["No", "Yes"])
            input_data[feature] = 1 if selected == "Yes" else 0

        else:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0)

# Predict button
st.markdown("---")
if st.button("🎯 Predict Churn"):
    try:
        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn. Probability: {proba:.2f}")
        else:
            st.success(f"✅ This customer is likely to stay. Probability: {1 - proba:.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
