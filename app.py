import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========== Load Model, Scaler, and Feature Names ==========
if not all(os.path.exists(f) for f in ['credit_risk_model.pkl', 'scaler.pkl', 'feature_names.pkl']):
    st.error("Required model files not found. Make sure 'credit_risk_model.pkl', 'scaler.pkl', and 'feature_names.pkl' are present.")
    st.stop()

model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('feature_names.pkl')  # list of column names used during training

# ========== Streamlit App Layout ==========
st.title("💳 Credit Risk Prediction App")
st.write("Enter applicant details to predict their credit risk:")

# ========== User Input ==========
age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_amount = st.number_input("Credit Amount", min_value=100, value=2000)
duration = st.number_input("Loan Duration (months)", min_value=6, value=24)
sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "no_info"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "no_info"])
purpose = st.selectbox("Purpose", [
    "radio/TV", "education", "furniture/equipment", "car",
    "business", "domestic appliances", "repairs", "vacation/others"
])

age_group = "Young" if age <= 30 else ("Middle-aged" if age <= 50 else "Senior")

# ========== Predict Button ==========
if st.button("Predict Credit Risk"):
    # Create input feature dictionary (one-hot encoded)
    input_dict = {
        'Age': [age],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Credit_per_month': [credit_amount / duration],
        'AgeGroup_Senior': [1 if age_group == "Senior" else 0],
        'AgeGroup_Young': [1 if age_group == "Young" else 0],
        'Sex_male': [1 if sex == "male" else 0],
        'Housing_own': [1 if housing == "own" else 0],
        'Housing_rent': [1 if housing == "rent" else 0],
        'Saving accounts_moderate': [1 if saving_accounts == "moderate" else 0],
        'Saving accounts_no_info': [1 if saving_accounts == "no_info" else 0],
        'Saving accounts_quite rich': [1 if saving_accounts == "quite rich" else 0],
        'Saving accounts_rich': [1 if saving_accounts == "rich" else 0],
        'Checking account_moderate': [1 if checking_account == "moderate" else 0],
        'Checking account_no_info': [1 if checking_account == "no_info" else 0],
        'Checking account_rich': [1 if checking_account == "rich" else 0],
        'Purpose_car': [1 if purpose == "car" else 0],
        'Purpose_domestic appliances': [1 if purpose == "domestic appliances" else 0],
        'Purpose_education': [1 if purpose == "education" else 0],
        'Purpose_furniture/equipment': [1 if purpose == "furniture/equipment" else 0],
        'Purpose_radio/TV': [1 if purpose == "radio/TV" else 0],
        'Purpose_repairs': [1 if purpose == "repairs" else 0],
        'Purpose_vacation/others': [1 if purpose == "vacation/others" else 0],
    }

    input_df = pd.DataFrame(input_dict)

    # Add any missing columns from model_features
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column order matches training
    input_df = input_df[model_features]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Output result
    if prediction == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("⚠️ Bad Credit Risk")

