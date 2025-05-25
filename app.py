import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, encoders, and feature names
pipeline = joblib.load("pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")
encoders = joblib.load("encoders.pkl")

# Column descriptions for tooltips
column_help = {
    "gender": "Gender of the customer (Male, Female)",
    "SeniorCitizen": "Whether the customer is a senior citizen (0 = No, 1 = Yes)",
    "Partner": "Whether the customer has a partner (Yes, No)",
    "Dependents": "Whether the customer has dependents (Yes, No)",
    "tenure": "Number of months the customer has stayed with the company",
    "PhoneService": "Whether the customer has a phone service (Yes, No)",
    "MultipleLines": "Whether the customer has multiple lines (Yes, No, No phone service)",
    "InternetService": "Customer’s internet service provider (DSL, Fiber optic, No)",
    "OnlineSecurity": "Whether the customer has online security (Yes, No, No internet service)",
    "OnlineBackup": "Whether the customer has online backup (Yes, No, No internet service)",
    "DeviceProtection": "Whether the customer has device protection (Yes, No, No internet service)",
    "TechSupport": "Whether the customer has tech support (Yes, No, No internet service)",
    "StreamingTV": "Whether the customer uses streaming TV (Yes, No, No internet service)",
    "StreamingMovies": "Whether the customer uses streaming movies (Yes, No, No internet service)",
    "Contract": "Type of contract (Month-to-month, One year, Two year)",
    "PaperlessBilling": "Whether billing is paperless (Yes, No)",
    "PaymentMethod": "Payment method (Electronic check, Mailed check, Bank transfer, Credit card)",
    "MonthlyCharges": "The amount charged to the customer monthly",
    "TotalCharges": "The total amount charged to the customer",
}

st.set_page_config(page_title="ChurnForecast", layout="centered")
st.title("🔍 ChurnForecast: Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# Optional: Display all column meanings
with st.expander("ℹ️ What do these fields mean?"):
    for col, desc in column_help.items():
        st.markdown(f"**{col}**: {desc}")

# Form input
input_data = {}
for feature in feature_names:
    help_text = column_help.get(feature, "")

    if feature in encoders:
        options = list(encoders[feature].classes_)
        selected_option = st.selectbox(f"{feature}", options, help=help_text)
        encoded_value = encoders[feature].transform([selected_option])[0]
        input_data[feature] = encoded_value
    else:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0, help=help_text)

# Prediction button
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
