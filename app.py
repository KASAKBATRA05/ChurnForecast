import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pipeline, feature names, and encoders
pipeline = joblib.load("pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="ChurnForecast", layout="centered")
st.title("🔍 ChurnForecast: Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# Input form
input_data = {}
for feature in feature_names:
    if feature in encoders:
        options = list(encoders[feature].classes_)
        selected_option = st.selectbox(f"{feature}", options)
        encoded_value = encoders[feature].transform([selected_option])[0]
        input_data[feature] = encoded_value
    else:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    try:
        input_df = input_df[feature_names]
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. Probability: {proba:.2f}")
        else:
            st.success(f"✅ The customer is likely to stay. Probability: {1 - proba:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
