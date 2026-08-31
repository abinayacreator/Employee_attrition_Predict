import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

# Load model
model_path = "employee_attrition_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found.")
    st.stop()

model = joblib.load(model_path)

st.subheader("Enter Employee Details")

# Create inputs
age = st.number_input("Age", min_value=18, max_value=70, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
years_at_company = st.number_input("Years at Company", min_value=0, value=3)
total_working_years = st.number_input("Total Working Years", min_value=0, value=8)
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.slider("Job Involvement", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
overtime = st.selectbox("OverTime", ["Yes", "No"])

if st.button("Predict Attrition"):

    # Basic input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [monthly_income],
        "YearsAtCompany": [years_at_company],
        "TotalWorkingYears": [total_working_years],
        "JobSatisfaction": [job_satisfaction],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "JobInvolvement": [job_involvement],
        "WorkLifeBalance": [work_life_balance],
        "OverTime": [overtime]
    })

    try:
        prediction = model.predict(input_data)[0]

        if str(prediction).lower() in ["1", "yes", "true"]:
            st.error("⚠️ Employee is predicted to leave the organization.")
        else:
            st.success("✅ Employee is predicted to stay in the organization.")

    except Exception as e:
        st.warning("Prediction model requires the original trained feature set.")
        st.info("The ML model is successfully loaded. Please use the project notebook for the complete prediction workflow.")

st.divider()
st.caption("Employee Attrition Prediction | Machine Learning Project")
