import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face Hub
model_path = hf_hub_download(repo_id="your-Leron7/loan-approval-model", filename="model.pkl")

# Load the model
with open(model_path, "rb") as file:
    model = joblib.load(file)

# App title
st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant's details to predict loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode inputs
def encode_inputs():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_val = 0 if education == "Graduate" else 1
    self_emp_val = 1 if self_employed == "Yes" else 0
    prop_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    return [gender_val, married_val, dependents_val, education_val, self_emp_val,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, prop_val]

# Prediction
if st.button("Predict Loan Approval"):
    input_data = np.array([encode_inputs()])
    prediction = model.predict(input_data)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.subheader(f"Prediction Result: {result}")