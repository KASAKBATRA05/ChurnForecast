# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model, encoders, and features
pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Churn Forecast – Customer Churn Prediction App")

st.markdown("Use the form below to predict if a customer is likely to churn.")

# Define binary numeric fields (not encoded by LabelEncoder)
binary_numeric_fields = ["Senior Citizen"]

# Two-column layout
col1, col2 = st.columns(2)
input_data = {}

for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        if feature in encoders:
            options = list(encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", options)
            encoded = encoders[feature].transform([selected])[0]
            input_data[feature] = encoded
        elif feature in binary_numeric_fields:
            input_data[feature] = st.selectbox(f"{feature}", options=[0, 1])
        else:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0)

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
        st.error(f"Prediction failed: {e}")









