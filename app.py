import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model pipeline, feature names list, and encoders dictionary
pipeline = joblib.load("pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")
encoders = joblib.load("encoders.pkl")

# Dictionary of column descriptions for tooltips and info
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

# Set Streamlit page config
st.set_page_config(page_title="ChurnForecast", layout="centered")

# App title and instructions
st.title("🔍 ChurnForecast: Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# Expandable section with feature explanations
with st.expander("ℹ️ What do these fields mean?"):
    for col, desc in column_help.items():
        st.markdown(f"**{col}**: {desc}")

# Use a form for inputs and submission
with st.form("input_form"):
    # Create two columns to place inputs side-by-side
    cols = st.columns(2)
    input_data = {}

    for i, feature in enumerate(feature_names):
        help_text = column_help.get(feature, "")
        col = cols[i % 2]  # Alternate between two columns

        # For categorical features, use dropdown with options from encoder
        if feature in encoders:
            options = list(encoders[feature].classes_)
            selected_option = col.selectbox(f"{feature}", options, help=help_text, key=feature)
            encoded_value = encoders[feature].transform([selected_option])[0]
            input_data[feature] = encoded_value
        else:
            # For numerical features, use number input with some sensible ranges
            min_val = 0.0
            max_val = 100000.0  # large max by default
            step = 1.0

            if feature == "tenure":
                max_val = 72  # typical max tenure in months
            elif feature in ["MonthlyCharges", "TotalCharges"]:
                step = 0.1  # monetary values, allow decimals

            input_data[feature] = col.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                step=step,
                help=help_text,
                key=feature
            )

    # Submit button inside the form
    submitted = st.form_submit_button("Predict Churn")

# After submission, do prediction and display result
if submitted:
    input_df = pd.DataFrame([input_data])  # Create a DataFrame for the model
    try:
        # Ensure columns are in the correct order for the pipeline
        input_df = input_df[feature_names]

        # Predict churn class and probability
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        # Show results with colored feedback
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. Probability: {proba:.2%}")
        else:
            st.success(f"✅ The customer is likely to stay. Probability: {(1 - proba):.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


