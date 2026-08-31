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

# Dataset
DATA_FILE = "employee_attrition.csv"

try:
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_FILE, encoding="latin1")

# Fix column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace('\ufeff', '', regex=False)
    .str.replace('Ã¯Â»Â¿', '', regex=False)
    .str.strip()
)

# Check target
if "Attrition" not in df.columns:
    st.error("Attrition column not found.")
    st.write(df.columns.tolist())
    st.stop()

st.success("Employee Attrition dataset loaded successfully!")

# Remove unnecessary columns
drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

X = df.drop(
    columns=["Attrition"] + [
        c for c in drop_cols if c in df.columns
    ],
    errors="ignore"
)

y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Remove invalid rows
valid = y.notna()

X = X.loc[valid]
y = y.loc[valid].astype(int)

# Encode categorical columns
X = pd.get_dummies(
    X,
    drop_first=True
)

X = X.fillna(0)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Model
model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

# Display
st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")

st.subheader("Employee Attrition Prediction")

age = st.number_input(
    "Age",
    18,
    70,
    30
)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    200000,
    5000
)

years_at_company = st.number_input(
    "Years At Company",
    0,
    50,
    3
)

total_working_years = st.number_input(
    "Total Working Years",
    0,
    50,
    8
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

if st.button("Predict Attrition"):

    input_df = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )

    values = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "TotalWorkingYears": total_working_years,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "WorkLifeBalance": work_life_balance
    }

    for col, value in values.items():
        if col in input_df.columns:
            input_df[col] = value

    if overtime == "Yes":
        for col in input_df.columns:
            if col == "OverTime_Yes":
                input_df[col] = 1

    result = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

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

st.divider()

st.caption(
    "Employee Attrition Prediction | Machine Learning Project"
    )
