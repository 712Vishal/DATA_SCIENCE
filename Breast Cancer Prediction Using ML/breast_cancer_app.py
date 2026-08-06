import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

FEATURE_NAMES = [
    'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
]

DATA_FILE = 'Y:/Data science/Breast Cancer Prediction Using ML/breast_cancer.csv'


@st.cache_resource
def load_and_train_model(filepath):
    """
    Loads data, cleans it, detects the target column,
    trains an SVC model, and returns the model and scaler.
    """
    try:
        st.info(f"Training model based on local '{filepath}'. This runs only once.")
        df = pd.read_csv(filepath)

        # --- Normalize column names ---
        df.columns = [c.strip().lower() for c in df.columns]

        # --- Clean the 'bare nuclei' column ---
        if 'bare nuclei' in df.columns:
            df = df[df['bare nuclei'] != '?']
            df['bare nuclei'] = pd.to_numeric(df['bare nuclei'])

        # Drop 'id' column if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # --- Detect target column automatically ---
        possible_targets = ['class', 'diagnosis', 'label', 'target']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            st.error(f"❌ No target column found. Columns in file: {df.columns.tolist()}")
            return None, None

        # --- Separate features and target ---
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # --- Map target values if needed (e.g., M/B → 4/2) ---
        if y.dtype == 'O':
            y = y.str.upper().map({'M': 4, 'B': 2}).fillna(y)

        # --- Scaling ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Model Training ---
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_scaled, y)

        return model, scaler

    except FileNotFoundError:
        st.error(f"❌ Error: The data file '{filepath}' was not found.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None


# --- Load or Train Model ---
model, scaler = load_and_train_model(DATA_FILE)

if model is None:
    st.stop()


# --- App Header ---
st.title("🔬 Advanced Breast Cancer Predictor")
st.markdown("---")

st.markdown("""
This app predicts the likelihood of a cell sample being **Benign (2)** or **Malignant (4)**  
based on the characteristics of the cell nuclei.  
Use the sliders in the sidebar to set values between **1 (normal)** and **10 (abnormal)**.
""")


# --- Sidebar Inputs ---
st.sidebar.header("🧬 Cell Sample Features (1–10)")

def user_input_features():
    data = {}
    for feature in FEATURE_NAMES:
        data[feature] = st.sidebar.slider(feature, 1, 10, 5, 1)
    return pd.DataFrame(data, index=['Input'])

df_input = user_input_features()

st.header("📋 User Input Features")
st.dataframe(df_input)
st.markdown("---")


# --- Prediction Logic ---
if st.button('Run Prediction', type="primary"):
    try:
        input_scaled = scaler.transform(df_input)
        prediction = model.predict(input_scaled)
        decision_score = model.decision_function(input_scaled)[0]

        prediction_result = "Malignant (4)" if prediction[0] == 4 else "Benign (2)"
        confidence_score = abs(decision_score)

        if prediction[0] == 4:
            color, emoji, summary = "red", "🚨", "The model suggests the sample is **Malignant**. Please consult a medical professional."
        else:
            color, emoji, summary = "green", "✅", "The model suggests the sample is **Benign**."

        st.subheader(f"{emoji} Prediction Result")
        st.markdown(f"""
            <div style='background-color:{color};padding:15px;border-radius:10px;color:white;'>
                <h3 style='margin-top:0;'>Result: {prediction_result}</h3>
                <p>Confidence Score (distance from boundary): <b>{confidence_score:.2f}</b></p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Summary:** {summary}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --- Footer ---
st.markdown("---")
st.caption("Model trained using Support Vector Classifier (SVC). For research and educational purposes only. Always consult a medical professional.")
