import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💼"
)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

# Find CSV automatically
csv_files = []

for file in os.listdir("."):
    if file.lower().endswith(".csv"):
        csv_files.append(file)

if not csv_files:
    st.error("CSV dataset not found.")
    st.stop()

df = pd.read_csv(csv_files[0])

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# Automatically find target column
possible_targets = [
    "Attrition",
    "attrition",
    "Employee_Attrition",
    "employee_attrition",
    "Left",
    "left",
    "Exited",
    "exited",
    "Turnover",
    "turnover"
]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

# If target is not found, use the last column
if target is None:
    target = df.columns[-1]

st.info(f"Target variable used: {target}")

# Remove duplicate rows
df = df.drop_duplicates()

# Separate X and y
X = df.drop(columns=[target])
y = df[target]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Convert target to numeric
if y.dtype == "object":
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.astype(str))
else:
    y = y.astype(int)

# Fill missing values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.success(f"Model Accuracy: {accuracy:.4f}")

# Prediction section
st.subheader("Employee Details")

input_data = {}

for col in X.columns:

    if X[col].dtype in ["int64", "float64"]:
        input_data[col] = st.number_input(
            col,
            value=float(X[col].median())
        )
    else:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

if st.button("Predict Attrition"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the organization.")
    else:
        st.success("✅ Employee is likely to stay in the organization.")

st.divider()
st.caption("Employee Attrition Prediction | Machine Learning Project")
