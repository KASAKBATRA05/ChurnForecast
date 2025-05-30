import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Churn Forecast – Customer Churn Prediction")

# Properly restrict binary numeric features
binary_numeric_fields = ["Senior Citizen"]

col1, col2 = st.columns(2)
input_data = {}

for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        if feature in binary_numeric_fields:
            input_data[feature] = st.selectbox(f"{feature}", options=[0, 1])
        elif feature in encoders:
            options = list(encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", options)
            input_data[feature] = encoders[feature].transform([selected])[0]
        else:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0)

st.markdown("---")

if st.button("🎯 Predict Churn"):
    input_df = pd.DataFrame([input_data])[feature_names]
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to churn – Probability: {proba:.2f}")
    else:
        st.success(f"✅ Likely to stay – Probability: {1 - proba:.2f}")


