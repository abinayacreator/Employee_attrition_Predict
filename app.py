import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💼",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

DATA_FILE = "employee_attrition.csv"

try:
    df = pd.read_csv(DATA_FILE, encoding="latin1")
except Exception as e:
    st.error("Unable to load employee_attrition.csv")
    st.write(e)
    st.stop()

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# ------------------------------------------------
# CHECK FOR CORRECT EMPLOYEE ATTRITION DATA
# ------------------------------------------------

required_columns = [
    "Age",
    "Attrition"
]

if not all(col in df.columns for col in required_columns):

    st.error(
        "The uploaded employee_attrition.csv is not the correct Employee Attrition dataset."
    )

    st.write("Columns found in the file:")
    st.write(df.columns.tolist())

    st.info(
        "Please upload the correct Employee Attrition CSV containing an 'Attrition' column."
    )

    st.stop()

# ------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------

df = df.drop_duplicates()

# Remove unnecessary columns
remove_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=[
        c for c in remove_columns
        if c in df.columns
    ],
    errors="ignore"
)

# Target
y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Remove invalid target rows
valid_rows = y.notna()

df = df.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

# Features
X = df.drop(
    columns=["Attrition"],
    errors="ignore"
)

# Convert categorical variables
X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Handle missing values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# MODEL
# ------------------------------------------------

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# ------------------------------------------------
# DISPLAY METRICS
# ------------------------------------------------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "Precision",
    f"{precision:.4f}"
)

col3.metric(
    "Recall",
    f"{recall:.4f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.4f}"
)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

st.subheader("Employee Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=5000
)

years_at_company = st.number_input(
    "Years At Company",
    min_value=0,
    max_value=50,
    value=3
)

total_working_years = st.number_input(
    "Total Working Years",
    min_value=0,
    max_value=50,
    value=8
)

job_satisfaction = st.slider(
    "Job Satisfaction",
    1,
    4,
    3
)

environment_satisfaction = st.slider(
    "Environment Satisfaction",
    1,
    4,
    3
)

job_involvement = st.slider(
    "Job Involvement",
    1,
    4,
    3
)

work_life_balance = st.slider(
    "Work Life Balance",
    1,
    4,
    3
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------

if st.button(
    "Predict Attrition",
    type="primary"
):

    # Start with all model features as zero
    input_data = pd.DataFrame(
        np.zeros(
            (1, len(X.columns))
        ),
        columns=X.columns
    )

    # Numerical features
    for col, value in {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "TotalWorkingYears": total_working_years,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "WorkLifeBalance": work_life_balance
    }.items():

        if col in input_data.columns:
            input_data[col] = value

    # OverTime
    if overtime == "Yes":

        overtime_columns = [
            c for c in X.columns
            if "OverTime_Yes" in c
        ]

        for col in overtime_columns:
            input_data[col] = 1

    # Prediction
    result = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if result == 1:

        st.error(
            "⚠️ Employee is likely to leave the organization."
        )

    else:

        st.success(
            "✅ Employee is likely to stay in the organization."
        )

    st.write(
        f"Attrition Probability: {probability:.2%}"
    )

# ------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------

with st.expander("Dataset Information"):

    st.write(
        "Dataset Shape:",
        df.shape
    )

    st.write(
        "Number of Features:",
        X.shape[1]
    )

    st.write(
        "Target Variable: Attrition"
)
