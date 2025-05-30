# app.py

import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Use this form to check if a customer is likely to churn.")

# ✅ These features are binary numeric (not encoded, but must be restricted)
binary_numeric_features = ["Senior Citizen"]

# 🧾 Collect user input
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        if feature in encoders:
            # Category features (encoded using LabelEncoder)
            options = list(encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", options)
            encoded = encoders[feature].transform([selected])[0]
            input_data[feature] = encoded
        elif feature in binary_numeric_features:
            # Binary numeric fields: restrict to 0 or 1
            selected = st.selectbox(f"{feature}", ["No", "Yes"])
            input_data[feature] = 1 if selected == "Yes" else 0
        else:
            # Numeric continuous fields
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0)

st.markdown("---")

# 🔮 Predict on button click
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



