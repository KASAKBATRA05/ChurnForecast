# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
pipeline = joblib.load("pipeline.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Churn Forecast", layout="wide")
st.title("📊 Churn Forecast – Customer Churn Prediction App")

tab1, tab2 = st.tabs(["🔮 Predict Churn", "📈 Model Performance"])

# -------- TAB 1: PREDICTION FORM -------- #
with tab1:
    st.markdown("### 🧾 Customer Information")

    col1, col2 = st.columns(2)
    input_data = {}

    for i, feature in enumerate(feature_names):
        # Alternate columns for cleaner layout
        with col1 if i % 2 == 0 else col2:
            if feature in encoders:
                options = list(encoders[feature].classes_)
                selected = st.selectbox(f"{feature}", options)
                encoded_value = encoders[feature].transform([selected])[0]
                input_data[feature] = encoded_value
            else:
                # Get valid range from training if available
                if feature == "Senior Citizen":
                    val = st.selectbox(f"{feature}", options=[0, 1])
                else:
                    val = st.number_input(f"{feature}", min_value=0.0)
                input_data[feature] = val

    st.markdown("---")

    if st.button("🎯 Predict"):
        try:
            input_df = pd.DataFrame([input_data])[feature_names]
            prediction = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ The customer is likely to churn! Probability: {proba:.2f}")
            else:
                st.success(f"✅ The customer is likely to stay. Probability: {1 - proba:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------- TAB 2: METRICS -------- #
with tab2:
    st.markdown("### 📊 Model Evaluation")
    st.write("Here are the performance metrics for the trained XGBoost model:")

    try:
        with open("classification_report.txt", "r") as f:
            report = f.read()
        st.code(report, language="text")
    except FileNotFoundError:
        st.warning("Model performance report not found. Please re-run `train_model.py` to regenerate.")





