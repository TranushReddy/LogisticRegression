import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Logistic Regression App", layout="wide")

st.title("Logistic Regression using  breast Cancer Dataset")
st.write("Upload a CSV dataset and train a Logistic Regression model automatically.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    # Load Dataset
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
        index=list(df.columns).index("diagnosis") if "diagnosis" in df.columns else 0,
    )

    # Encode target column if it's non-numeric
    if df[target_column].dtype == object or str(df[target_column].dtype).startswith(
        "category"
    ):
        if set(df[target_column].unique()) <= {"M", "B"}:
            df[target_column] = (df[target_column] == "M").astype(int)
        else:
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])

    # Compute correlation matrix and get absolute correlation with chosen target
    corr = df.corr()
    cor_target = abs(corr[target_column])

    # Select highly correlated features (threshold = 0.2)
    relevant_features = cor_target[cor_target > 0.2]

    # Collect the names of the features
    names = relevant_features.index.tolist()

    # Drop the target variable from the results if present
    if target_column in names:
        names.remove(target_column)

    # Display the selected features
    st.write("Selected features based on correlation > 0.2:", names)

    # ---------------- FEATURES & TARGET ----------------
    if len(names) == 0:
        st.error(
            "No features selected based on the correlation threshold. Try a lower threshold or choose a different target column."
        )
        st.stop()

    X = df[names]
    y = df[target_column]

    # ---------------- ENCODE CATEGORICAL FEATURES ----------------
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    st.subheader("Feature Columns")
    st.write(X.columns.tolist())

    # ---------------- TRAIN TEST SPLIT ----------------
    test_size = st.slider(
        "Select Test Size", min_value=0.1, max_value=0.5, value=0.3, step=0.05
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)

    # ---------------- MODEL TRAINING ----------------
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)

    # ---------------- EVALUATION ----------------
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")

    st.write(f"Accuracy Score: {accuracy:.4f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(25, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ---------------- COEFFICIENTS ----------------
    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})

    st.dataframe(coef_df)

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig2, ax2 = plt.subplots(figsize=(30, 15))

    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)

    st.pyplot(fig2)

    # ---------------- USER INPUT PREDICTION ----------------
    st.subheader("Make Custom Prediction")

    user_input = {}

    for column in X.columns:
        user_input[column] = st.number_input(
            f"Enter value for {column}", value=float(X[column].mean())
        )

    input_df = pd.DataFrame([user_input])

    # Ensure columns order and types match training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_df = input_df.astype(float)

    # Scale the input using the same scaler as training
    input_scaled = Scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Class: {prediction[0]}")

else:
    st.info("Please upload a CSV dataset to continue.")
