import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- MOCKING: This simulates loading the trained model and preprocessing artifacts ---
# In a real deployment, these artifacts would be saved during training and loaded here.

# MOCK: Trained XGBoost Model (Placeholder)
# We provide a dummy prediction function for demonstration.
class MockXGBClassifier:
    """Simulates the trained model for demonstration purposes."""
    def predict(self, X):
        # The Liver dataset target is usually 1 (Liver Disease) or 2 (No Disease).
        # We will map it to 1 (Disease) and 0 (No Disease).
        
        # Simple dummy logic: predicts "Disease" if Total_Bilirubin (index 1 after scaling)
        # is high and Age (index 0) is over 50 (approximately 0.5 after mock scaling).
        age_scaled = X[0, 0]
        bilirubin_scaled = X[0, 1] 
        
        # Arbitrary decision boundary for mock prediction
        if (age_scaled > 0.5) or (bilirubin_scaled > 1.0):
            return np.array([1]) # 1 for Liver Disease
        else:
            return np.array([0]) # 0 for No Liver Disease

MOCK_MODEL = MockXGBClassifier()

# MOCK: Preprocessing Statistics (Inferred from common ILPD dataset statistics)
# These mock parameters ensure the scaling step is executable.
# Features: Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
#           Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, 
#           Albumin, Albumin_and_Globulin_Ratio
NUMERICAL_COLS = [
    'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
    'Albumin', 'Albumin_and_Globulin_Ratio'
]
GENDER_COL = 'Gender'

# Mock means for the 9 numerical features (Used for scaling and imputation of the ratio)
MOCK_MEANS = np.array([
    44.7,  1.7,  0.8, 290.0, 80.0, 100.0,  6.4,  3.1, 0.94 
])

# Mock standard deviations for the 9 numerical features
MOCK_SCALES = np.array([
    12.4,  3.5,  1.6, 209.0, 182.0, 269.0,  1.0,  0.8, 0.35 
])

# MOCK: Standard Scaler (Simulates the scaler fitted on the training data)
class MockStandardScaler:
    """Simulates a fitted StandardScaler with mock mean and scale."""
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_

MOCK_SCALER = MockStandardScaler(MOCK_MEANS, MOCK_SCALES)

# Final feature order for the model input (9 scaled numerical + 1 binary gender)
FINAL_FEATURE_ORDER = NUMERICAL_COLS + [GENDER_COL]


# --- Preprocessing Function ---
def preprocess_input(input_data):
    """
    Applies the full preprocessing pipeline (Imputation, Encoding, Scaling) 
    to the single input sample.
    """
    input_df = pd.DataFrame([input_data])
    
    # 1. Handle Missing Values (Mock imputation for Albumin_and_Globulin_Ratio)
    # The UI makes all fields mandatory, but we ensure the ratio field is filled 
    # if it were somehow missing, using the mock mean.
    if input_df['Albumin_and_Globulin_Ratio'].iloc[0] is None:
        input_df['Albumin_and_Globulin_Ratio'] = MOCK_MEANS[-1]
    
    # 2. Gender Encoding (Male=1, Female=0)
    # This is a common approach for the ILPD dataset where Male=1 is often used
    # to represent the more prevalent class for liver disease.
    input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # 3. Separate Numerical and Categorical Features
    numerical_df = input_df[NUMERICAL_COLS].astype(float)
    gender_df = input_df[[GENDER_COL]]
    
    # 4. Feature Scaling for Numerical Features
    numerical_scaled = MOCK_SCALER.transform(numerical_df.values)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=NUMERICAL_COLS)
    
    # 5. Recombine and ensure final column order
    final_input_df = pd.concat([numerical_scaled_df, gender_df], axis=1)
    
    # Ensure the final array has the exact column order the model expects
    X_final = final_input_df[FINAL_FEATURE_ORDER].values

    return X_final.reshape(1, -1) # Ensure it's 2D array


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Liver Disease Prediction (ML Model)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}
.main-header {
    font-size: 2.5em;
    font-weight: 700;
    color: #006400; /* DarkGreen */
    text-align: center;
    padding-bottom: 20px;
}
.stButton>button {
    background-color: #3CB371; /* MediumSeaGreen */
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
}
.stButton>button:hover {
    background-color: #2E8B57; /* SeaGreen */
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
}
.prediction-box {
    padding: 20px;
    border-radius: 0.75rem;
    text-align: center;
    font-size: 1.5em;
    font-weight: 600;
    margin-top: 30px;
    border: 2px solid;
}
.disease {
    background-color: #fff0f0;
    border-color: #cd5c5c;
    color: #cd5c5c;
}
.no-disease {
    background-color: #f0fff0;
    border-color: #3cb371;
    color: #3cb371;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Liver Disease Prediction Model</div>', unsafe_allow_html=True)
st.write("Enter the patient's **Demographics** in the sidebar and **Lab Parameters** below to predict the presence of Liver Disease (simulated XGBoost model).")

# --- Sidebar Inputs (Demographics) ---
with st.sidebar:
    st.header("Patient Demographics")
    
    # Age
    age = st.number_input("Age (years)", min_value=1, max_value=90, value=40, help="Age of the patient")
    
    # Gender
    gender = st.selectbox("Gender", options=['Male', 'Female'], index=0, help="Patient's gender")
    
    st.markdown("---")
    st.info("Fill out the lab values in the main panel and click 'Predict' when ready.")


# --- Input Form using Columns (Main Panel) ---
with st.form("liver_prediction_form"):
    
    st.subheader("Laboratory Values")

    # Group 1: Enzymes (3 columns)
    st.markdown("##### Liver Enzymes")
    col4, col5, col6 = st.columns(3)

    alp = col4.number_input("Alkaline Phosphotase (ALP)", min_value=65.0, max_value=2000.0, value=180.0, step=10.0, help="Normal range is 65-300 IU/L")
    alt = col5.number_input("Alamine Aminotransferase (ALT)", min_value=10.0, max_value=2000.0, value=40.0, step=5.0, help="Also known as SGPT. Normal range is 10-50 U/L")
    ast = col6.number_input("Aspartate Aminotransferase (AST)", min_value=10.0, max_value=2000.0, value=40.0, step=5.0, help="Also known as SGOT. Normal range is 10-40 U/L")
    
    st.markdown("---")
    
    # Group 2: Bilirubin (2 columns)
    st.markdown("##### Bilirubin Levels")
    col7, col8 = st.columns(2)
    
    t_bil = col7.number_input("Total Bilirubin", min_value=0.2, max_value=75.0, value=1.0, step=0.1, format="%.1f", help="Normal range is 0.2-1.2 mg/dL")
    d_bil = col8.number_input("Direct Bilirubin", min_value=0.1, max_value=20.0, value=0.2, step=0.1, format="%.1f", help="Normal range is 0.1-0.3 mg/dL")
    
    st.markdown("---")

    # Group 3: Proteins (3 columns)
    st.markdown("##### Protein Markers")
    col9, col10, col11 = st.columns(3)
    
    t_prot = col9.number_input("Total Proteins", min_value=3.0, max_value=10.0, value=6.5, step=0.1, format="%.1f", help="Normal range is 6.0-8.3 g/dL")
    alb = col10.number_input("Albumin", min_value=1.5, max_value=5.5, value=3.5, step=0.1, format="%.1f", help="Normal range is 3.5-5.0 g/dL")
    a_g_ratio = col11.number_input("Albumin and Globulin Ratio", min_value=0.2, max_value=3.0, value=0.9, step=0.05, format="%.2f", help="Calculated as Albumin / (Total Proteins - Albumin). Normal range is 1.0-2.0")

    st.markdown("---")
    submitted = st.form_submit_button("Predict Liver Disease Risk")

    if submitted:
        # 1. Gather all inputs into a dictionary
        input_data = {
            'Age': age, 
            'Gender': gender, 
            'Total_Bilirubin': t_bil, 
            'Direct_Bilirubin': d_bil, 
            'Alkaline_Phosphotase': alp, 
            'Alamine_Aminotransferase': alt, 
            'Aspartate_Aminotransferase': ast, 
            'Total_Protiens': t_prot, 
            'Albumin': alb, 
            'Albumin_and_Globulin_Ratio': a_g_ratio
        }

        try:
            # 2. Preprocess the input data
            final_features = preprocess_input(input_data)
            
            # 3. Make Prediction
            prediction = MOCK_MODEL.predict(final_features)
            
            # 4. Display Result
            if prediction[0] == 1:
                result_text = "PREDICTION: HIGH RISK OF LIVER DISEASE"
                result_style = "disease"
                st.markdown(f'<div class="prediction-box {result_style}">🔴 {result_text}</div>', unsafe_allow_html=True)
                st.error("This result suggests a high risk of Liver Disease. Please consult a medical professional for diagnosis and treatment.")
            else:
                result_text = "PREDICTION: LOW RISK OF LIVER DISEASE"
                result_style = "no-disease"
                st.markdown(f'<div class="prediction-box {result_style}">🟢 {result_text}</div>', unsafe_allow_html=True)
                st.success("The model predicts a low risk of Liver Disease based on the provided inputs. Continue with healthy lifestyle and regular monitoring.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure all input fields are properly filled and try again.")

# Footer
st.markdown("""
---
<p style='text-align: center; font-size: 0.8em; color: #777;'>
This application is for educational and demonstrative purposes only. Model performance is simulated. 
For medical use, ensure the model, scaler, and all preprocessing artifacts are correctly loaded 
from the trained pipeline and validated by an expert.
</p>
""", unsafe_allow_html=True)
