import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the organization.")

# --------------------------------------------------
# FIND DATASET
# --------------------------------------------------

csv_files = [
    f for f in os.listdir(".")
    if f.lower().endswith(".csv")
]

if len(csv_files) == 0:
    st.error("No CSV dataset found in the project folder.")
    st.stop()

# Try each CSV until a valid one is found
df = None
used_file = None

for file in csv_files:

    try:
        temp = pd.read_csv(file, encoding="utf-8")
        df = temp
        used_file = file
        break

    except UnicodeDecodeError:

        try:
            temp = pd.read_csv(file, encoding="latin1")
            df = temp
            used_file = file
            break

        except Exception:
            continue

    except Exception:
        continue

if df is None:
    st.error("Unable to read the CSV dataset.")
    st.stop()

st.success(f"Dataset loaded: {used_file}")

# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

# --------------------------------------------------
# FIND TARGET COLUMN
# --------------------------------------------------

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

# If Attrition is not found, search column names
if target is None:

    for col in df.columns:
        if "attrition" in col.lower():
            target = col
            break

# Last fallback
if target is None:
    target = df.columns[-1]

st.info(f"Target column: {target}")

# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

df = df.drop_duplicates()

# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=[target])
y = df[target]

# --------------------------------------------------
# TARGET ENCODING
# --------------------------------------------------

if y.dtype == "object":

    y = y.astype(str).str.strip()

    # Common Yes/No format
    y = y.replace({
        "Yes": 1,
        "No": 0,
        "YES": 1,
        "NO": 0,
        "yes": 1,
        "no": 0
    })

    # If still categorical, encode automatically
    if not pd.api.types.is_numeric_dtype(y):

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

else:
    y = pd.to_numeric(y, errors="coerce")

# Remove rows with invalid target
valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# --------------------------------------------------
# REMOVE UNNECESSARY COLUMNS
# --------------------------------------------------

remove_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

X = X.drop(
    columns=[c for c in remove_columns if c in X.columns],
    errors="ignore"
)

# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Fill missing values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("Precision", f"{precision:.4f}")
c3.metric("Recall", f"{recall:.4f}")
c4.metric("F1 Score", f"{f1:.4f}")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Employee Prediction")

input_data = {}

for col in X.columns:

    input_data[col] = float(
        X[col].median()
    )

input_df = pd.DataFrame(
    [input_data]
)

input_df = input_df[X.columns]

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Attrition"):

    result = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

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

# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------

with st.expander("Dataset Information"):

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )

    st.write(
        "Features used:",
        list(X.columns)
    )

st.divider()

st.caption(
    "Employee Attrition Prediction | Machine Learning Project"
        )
