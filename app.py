st.set_page_config(page_title="ChurnForecast", layout="centered")
st.title("🔍 ChurnForecast: Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

with st.expander("ℹ️ What do these fields mean?"):
    for col, desc in column_help.items():
        st.markdown(f"**{col}**: {desc}")

with st.form("input_form"):
    cols = st.columns(2)  # 2 columns for better layout

    input_data = {}
    for i, feature in enumerate(feature_names):
        help_text = column_help.get(feature, "")
        col = cols[i % 2]  # alternate columns

        if feature in encoders:
            options = list(encoders[feature].classes_)
            selected_option = col.selectbox(f"{feature}", options, help=help_text, key=feature)
            encoded_value = encoders[feature].transform([selected_option])[0]
            input_data[feature] = encoded_value
        else:
            # Customize min/max for known features:
            min_val = 0.0
            max_val = 100000.0
            step = 1.0
            if feature == "tenure":
                max_val = 72  # Example max tenure in months
            if feature == "MonthlyCharges" or feature == "TotalCharges":
                step = 0.1
            input_data[feature] = col.number_input(f"{feature}", min_value=min_val, max_value=max_val, step=step, help=help_text, key=feature)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([input_data])
    try:
        input_df = input_df[feature_names]
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. Probability: {proba:.2%}")
        else:
            st.success(f"✅ The customer is likely to stay. Probability: {(1 - proba):.2%}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

