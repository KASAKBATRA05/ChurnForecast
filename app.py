# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Predictor")
st.write("Enter customer details below:")

# Create input form
input_data = {}
for feature in feature_names:
    if feature in encoders:
        options = list(encoders[feature].classes_)
        selected = st.selectbox(f"{feature}", options)
        encoded_value = encoders[feature].transform([selected])[0]
        input_data[feature] = encoded_value
    else:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0)

# Predict button
if st.button("Predict Churn"):
    try:
        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn. Probability: {proba:.2f}")
        else:
            st.success(f"✅ This customer is likely to stay. Probability: {1 - proba:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")



