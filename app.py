import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

# -----------------------------
# Load Dataset
# -----------------------------

DATA_PATH = "employee_attrition.csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found: employee_attrition.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Check Target
# -----------------------------

if "Attrition" not in df.columns:
    st.error("Attrition column not found in the dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -----------------------------
# Preprocessing
# -----------------------------

df = df.drop_duplicates()

# Remove unnecessary columns if they exist
drop_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)

# Target encoding
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

df = df.dropna(subset=["Attrition"])

# Separate X and y
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Fill missing values
X = X.fillna(0)

# -----------------------------
# Train Model
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# -----------------------------
# Display Model Accuracy
# -----------------------------

st.success(
    f"Model Accuracy: {accuracy:.4f}"
)

# -----------------------------
# User Inputs
# -----------------------------

st.subheader("Enter Employee Details")

input_values = {}

for column in X.columns:

    if "_" in column:

        input_values[column] = 0

    else:

        default_value = float(
            X[column].median()
        )

        input_values[column] = st.number_input(
            column,
            value=default_value
        )

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Attrition"):

    input_df = pd.DataFrame(
        [input_values]
    )

    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

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
