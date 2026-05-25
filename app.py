import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Logistic Regression App", layout="wide")

st.title("Logistic Regression using Breast Cancer Dataset")
st.write("Upload a CSV dataset and train a Logistic Regression model automatically.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    # ---------------- LOAD DATASET ----------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATASET INFO ----------------
    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # ---------------- MISSING VALUES ----------------
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Remove Missing Values
    df = df.dropna()

    # ---------------- TARGET COLUMN ----------------
    st.subheader("Select Target Column")

    target_column = st.selectbox(
        "Choose the target column",
        df.columns,
        index=list(df.columns).index("diagnosis")
        if "diagnosis" in df.columns
        else 0,
    )

    # ---------------- ENCODE TARGET COLUMN ----------------
    if (
        df[target_column].dtype == object
        or str(df[target_column].dtype).startswith("category")
    ):

        # Special handling for Breast Cancer dataset
        if set(df[target_column].unique()) <= {"M", "B"}:
            df[target_column] = (df[target_column] == "M").astype(int)

        else:
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column].astype(str))

    # ---------------- ENCODE ALL CATEGORICAL COLUMNS ----------------
    df_encoded = df.copy()

    for col in df_encoded.columns:

        if df_encoded[col].dtype == object:

            le = LabelEncoder()

            df_encoded[col] = le.fit_transform(
                df_encoded[col].astype(str)
            )

    # ---------------- CORRELATION ----------------
    corr = df_encoded.corr()

    cor_target = abs(corr[target_column])

    # Select highly correlated features
    relevant_features = cor_target[cor_target > 0.2]

    names = relevant_features.index.tolist()

    # Remove target column
    if target_column in names:
        names.remove(target_column)

    st.subheader("Selected Features")
    st.write(names)

    # ---------------- CHECK FEATURES ----------------
    if len(names) == 0:

        st.error(
            "No features selected based on correlation threshold."
        )

        st.stop()

    # ---------------- FEATURES & TARGET ----------------
    X = df[names]
    y = df[target_column]

    # ---------------- ENCODE FEATURE COLUMNS ----------------
    X = pd.get_dummies(X, drop_first=True)

    # ---------------- FEATURE COLUMNS ----------------
    st.subheader("Feature Columns")
    st.write(X.columns.tolist())

    # ---------------- TRAIN TEST SPLIT ----------------
    test_size = st.slider(
        "Select Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # ---------------- FEATURE SCALING ----------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ---------------- MODEL TRAINING ----------------
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)

    # ---------------- MODEL PERFORMANCE ----------------
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")

    st.write(f"Accuracy Score: {accuracy:.4f}")

    # ---------------- CLASSIFICATION REPORT ----------------
    st.subheader("Classification Report")

    st.text(classification_report(y_test, y_pred))

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ---------------- MODEL COEFFICIENTS ----------------
    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Coefficient": model.coef_[0],
        }
    )

    st.dataframe(coef_df)

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("Correlation Heatmap")

    numeric_df = df_encoded.select_dtypes(include=np.number)

    fig2, ax2 = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2,
    )

    st.pyplot(fig2)

    # ---------------- CUSTOM PREDICTION ----------------
    st.subheader("Make Custom Prediction")

    user_input = {}

    for column in X.columns:

        user_input[column] = st.number_input(
            f"Enter value for {column}",
            value=float(X[column].mean()),
        )

    input_df = pd.DataFrame([user_input])

    # Ensure same order
    input_df = input_df.reindex(
        columns=X.columns,
        fill_value=0,
    )

    input_df = input_df.astype(float)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    prediction_proba = model.predict_proba(input_scaled)

    st.success(f"Predicted Class: {prediction[0]}")

    st.subheader("Prediction Probability")

    st.write(prediction_proba)

else:

    st.info("Please upload a CSV dataset to continue.")
