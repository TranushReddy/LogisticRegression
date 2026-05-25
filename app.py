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
st.set_page_config(
    page_title="Logistic Regression App",
    layout="wide"
)

st.title("Logistic Regression ML App")

st.write(
    "Upload a CSV dataset and train a Logistic Regression model automatically."
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATA INFO ----------------
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
        "Choose Target Column",
        df.columns
    )

    # ---------------- COPY DATA ----------------
    df_encoded = df.copy()

    # ---------------- ENCODE ALL OBJECT COLUMNS ----------------
    le = LabelEncoder()

    for col in df_encoded.columns:

        # Convert all non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):

            df_encoded[col] = le.fit_transform(
                df_encoded[col].astype(str)
            )

    # ---------------- CORRELATION ----------------
    numeric_df = df_encoded.select_dtypes(include=[np.number])

    corr = numeric_df.corr(numeric_only=True)

    # Check target exists
    if target_column not in corr.columns:

        st.error(
            "Target column could not be converted properly."
        )

        st.stop()

    cor_target = abs(corr[target_column])

    # Feature Selection
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
            "No features selected. Try another target column."
        )

        st.stop()

    # ---------------- FEATURES & TARGET ----------------
    X = df_encoded[names]

    y = df_encoded[target_column]

    # ---------------- TRAIN TEST SPLIT ----------------
    test_size = st.slider(
        "Select Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    # ---------------- SCALING ----------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ---------------- MODEL ----------------
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)

    # ---------------- ACCURACY ----------------
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")

    st.success(f"Accuracy: {accuracy:.4f}")

    # ---------------- CLASSIFICATION REPORT ----------------
    st.subheader("Classification Report")

    st.text(
        classification_report(y_test, y_pred)
    )

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ---------------- COEFFICIENTS ----------------
    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame({
        "Feature": names,
        "Coefficient": model.coef_[0]
    })

    st.dataframe(coef_df)

    # ---------------- HEATMAP ----------------
    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)

    # ---------------- CUSTOM PREDICTION ----------------
    st.subheader("Custom Prediction")

    user_input = {}

    for column in names:

        user_input[column] = st.number_input(
            f"Enter value for {column}",
            value=float(X[column].mean())
        )

    input_df = pd.DataFrame([user_input])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    st.success(f"Predicted Class: {prediction[0]}")

    st.write("Prediction Probability:")

    st.write(probability)

else:

    st.info("Please upload a CSV dataset.")
