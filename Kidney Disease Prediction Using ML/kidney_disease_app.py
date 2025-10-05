import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# --- MOCKING: This section simulates loading the trained model and preprocessing artifacts ---
# In a real deployment, these artifacts (model, scaler, encoder maps) would be saved
# during training and loaded here.

# MOCK: Trained XGBoost Model (Placeholder)
# We initialize a model but will provide a dummy prediction function for demonstration.
# In a real app, you would load it: with open('xgb_model.pkl', 'rb') as f: model = pickle.load(f)
class MockXGBClassifier:
    """Simulates the trained model for demonstration purposes."""
    def predict(self, X):
        # A simple dummy prediction: predicts 'ckd' (Chronic Kidney Disease) if the age is over 50
        # and 'notckd' otherwise. This is purely for demonstration.
        age = X[0, 0] * 100 # Reverse scaling of age for mock logic (Approximate)
        if age > 50:
            return np.array([0]) # 0 for ckd (assuming 0=ckd, 1=notckd)
        else:
            return np.array([1]) # 1 for notckd

MOCK_MODEL = MockXGBClassifier()

# MOCK: Preprocessing Statistics (Inferred from common ML practices and the notebook)
# These would typically be calculated from the training data.
MOCK_MEANS = {
    'age': 52.0, 'bp': 76.0, 'bgr': 140.0, 'bu': 55.0, 'sc': 3.0, 'sod': 138.0, 
    'pot': 4.5, 'hemo': 12.5, 'pcv': 40, 'wc': 7900, 'rc': 4.6
}

MOCK_MODES = {
    'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent', 
    'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
}

# MOCK: Standard Scaler (Simulates the scaler fitted on the training data)
class MockStandardScaler:
    """Simulates a fitted StandardScaler with mock mean and scale."""
    def __init__(self, n_features):
        # Mocking mean and std deviation (scale) for the 11 continuous features
        self.mean_ = np.array(list(MOCK_MEANS.values()))
        # Arbitrary standard deviation for mock scaling
        self.scale_ = np.array([15, 10, 60, 40, 5, 4, 1, 2.5, 5, 2500, 1]) 
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_

MOCK_SCALER = MockStandardScaler(n_features=11)

# MOCK: Feature lists (Based on the notebook's final feature set structure)
# This assumes the order of features after one-hot encoding categorical variables.
# This order is CRITICAL for the model prediction.
CONTINUOUS_COLS = list(MOCK_MEANS.keys())
CATEGORICAL_COLS = list(MOCK_MODES.keys())
ORDINAL_COLS = ['sg', 'al', 'su'] # These are treated as categorical selections in the UI

# Final feature order *after* preprocessing for the model input
FINAL_FEATURE_ORDER = [
    'age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'sg_1.005', 'sg_1.010', 'sg_1.015', 'sg_1.020', 'sg_1.025', 
    'al_1', 'al_2', 'al_3', 'al_4', 'al_5', 
    'su_1', 'su_2', 'su_3', 'su_4', 'su_5',
    'rbc_normal', 'rbc_abnormal', 
    'pc_normal', 'pc_abnormal', 
    'pcc_present', 'pcc_notpresent', 
    'ba_present', 'ba_notpresent', 
    'htn_yes', 'htn_no', 
    'dm_yes', 'dm_no', 
    'cad_yes', 'cad_no', 
    'appet_good', 'appet_poor', 
    'pe_yes', 'pe_no', 
    'ane_yes', 'ane_no'
]


# --- Preprocessing Function ---
def preprocess_input(input_data):
    """
    Applies the full preprocessing pipeline (Imputation, Encoding, Scaling) 
    to the single input sample.
    """
    # 1. Convert input dictionary to a DataFrame (required for pandas operations)
    input_df = pd.DataFrame([input_data])
    
    # 2. Imputation (The Streamlit app makes all fields mandatory, but we can fill
    #    'sg', 'al', 'su' which are categorical but use numeric values in the dataset)
    #    Since all UI fields require input, imputation is technically not needed here, 
    #    but we use the MOCK_MEANS/MODES for consistency if the data was incomplete.
    
    # 3. Handling Ordinal/Categorical Columns (sg, al, su are numeric but treated as ordinal)
    #    We convert them to strings for consistent one-hot encoding later.
    for col in ORDINAL_COLS:
        input_df[col] = input_df[col].astype(str).str.strip()

    # 4. Separate Continuous and Categorical Features
    continuous_df = input_df[CONTINUOUS_COLS]
    categorical_df = input_df[CATEGORICAL_COLS + ORDINAL_COLS]
    
    # 5. One-Hot Encoding for Categorical Features
    # This step is crucial. We must ensure ALL possible categories are represented.
    encoded_df = pd.get_dummies(categorical_df, drop_first=False)
    
    # Manually re-align columns for the model input
    # Create a full template DataFrame with all expected encoded columns initialized to 0
    full_template = pd.DataFrame(0, index=[0], columns=FINAL_FEATURE_ORDER)

    # Place the continuous values first
    for col in CONTINUOUS_COLS:
        full_template.loc[0, col] = continuous_df.loc[0, col]
        
    # Overwrite the one-hot encoded columns where the user input a value
    for col in encoded_df.columns:
        # Map categories to the full feature name (e.g., 'rbc_normal' from 'rbc', 'normal')
        # We assume the notebook used an encoding similar to: column_value
        if col in FINAL_FEATURE_ORDER:
             full_template.loc[0, col] = encoded_df.loc[0, col]
        else: # Handle the case where category names were slightly different 
              # or need explicit mapping (e.g. sg=1.005 -> sg_1.005)
            pass

    # Drop the continuous columns from the final feature set for now, 
    # as we will scale them separately.
    feature_cols_no_cont = [col for col in FINAL_FEATURE_ORDER if col not in CONTINUOUS_COLS]
    final_encoded_df = full_template[feature_cols_no_cont]
    
    # 6. Feature Scaling for Continuous Features
    continuous_scaled = MOCK_SCALER.transform(continuous_df.values)
    continuous_scaled_df = pd.DataFrame(continuous_scaled, columns=CONTINUOUS_COLS)
    
    # 7. Recombine and ensure final column order
    final_input_df = pd.concat([continuous_scaled_df, final_encoded_df], axis=1)
    
    # Ensure the final array has the exact column order the model expects
    X_final = final_input_df[FINAL_FEATURE_ORDER].values

    return X_final

# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Kidney Disease Prediction (ML Model)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics (using Tailwind-like styling)
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}
.main-header {
    font-size: 2.5em;
    font-weight: 700;
    color: #f7f9fc; /* Indigo/Deep Purple */
    text-align: center;
    padding-bottom: 20px;
}
.stButton>button {
    background-color: #8A2BE2; /* BlueViolet */
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
}
.stButton>button:hover {
    background-color: #6A5ACD; /* SlateBlue */
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
.ckd {
    background-color: #ffe5e5;
    border-color: #d9534f;
    color: #d9534f;
}
.notckd {
    background-color: #e5ffe5;
    border-color: #5cb85c;
    color: #5cb85c;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Chronic Kidney Disease Prediction Developed By Vishal </div>', unsafe_allow_html=True)
st.write("Enter the patient's clinical parameters below to predict the presence of Chronic Kidney Disease using a machine learning model (simulated XGBoost).")

# --- Input Form using Columns ---
with st.form("kidney_prediction_form"):
    
    st.subheader("I. Clinical/Physical Parameters")
    col1, col2, col3, col4 = st.columns(4)

    age = col1.number_input("Age (years)", min_value=1, max_value=100, value=45, help="Age of the patient")
    bp = col2.number_input("Blood Pressure (mm/Hg)", min_value=60.0, max_value=180.0, value=80.0, step=5.0, help="Systolic blood pressure")
    
    sg = col3.selectbox("Specific Gravity (sg)", options=['1.025', '1.020', '1.015', '1.010', '1.005'], index=2, help="Urine specific gravity")
    al = col4.selectbox("Albumin (al) in Urine", options=['0', '1', '2', '3', '4', '5'], index=0, help="Albumin level (0=normal to 5=highest)")
    su = col1.selectbox("Sugar (su) in Urine", options=['0', '1', '2', '3', '4', '5'], index=0, help="Sugar level (0=normal to 5=highest)")
    
    htn = col2.selectbox("Hypertension (htn)", options=['no', 'yes'], index=0, help="Presence of high blood pressure")
    dm = col3.selectbox("Diabetes Mellitus (dm)", options=['no', 'yes'], index=0, help="Presence of diabetes")
    cad = col4.selectbox("Coronary Artery Disease (cad)", options=['no', 'yes'], index=0, help="Presence of coronary artery disease")
    
    st.markdown("---")
    st.subheader("II. Laboratory Values")
    col5, col6, col7, col8 = st.columns(4)
    
    bgr = col5.number_input("Blood Glucose Random (mgs/dl)", min_value=70.0, max_value=490.0, value=120.0, step=10.0, help="Random blood glucose level")
    bu = col6.number_input("Blood Urea (mgs/dl)", min_value=10.0, max_value=390.0, value=40.0, step=5.0, help="Level of urea in the blood")
    sc = col7.number_input("Serum Creatinine (mgs/dl)", min_value=0.5, max_value=70.0, value=1.0, step=0.1, help="Creatinine level in the serum")
    sod = col8.number_input("Sodium (mEq/L)", min_value=111.0, max_value=150.0, value=135.0, step=1.0, help="Sodium level")
    
    col9, col10, col11, col12 = st.columns(4)
    pot = col9.number_input("Potassium (mEq/L)", min_value=2.5, max_value=15.0, value=4.0, step=0.1, help="Potassium level")
    hemo = col10.number_input("Haemoglobin (gms)", min_value=3.0, max_value=18.0, value=14.0, step=0.5, help="Haemoglobin level")
    pcv = col11.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60, value=45, step=1, help="Volume percentage of red blood cells")
    wc = col12.number_input("White Blood Cell Count (cells/cmm)", min_value=2200, max_value=26400, value=7500, step=500, help="White blood cell count")
    
    col13, col14, col15, col16 = st.columns(4)
    rc = col13.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=5.0, step=0.1, help="Red blood cell count")

    st.markdown("---")
    st.subheader("III. Urine and Other Observations")
    col17, col18, col19, col20 = st.columns(4)

    rbc = col17.selectbox("Red Blood Cell (rbc)", options=['normal', 'abnormal'], index=0, help="Presence of red blood cells in urine")
    pc = col18.selectbox("Pus Cell (pc)", options=['normal', 'abnormal'], index=0, help="Presence of pus cells in urine")
    pcc = col19.selectbox("Pus Cell Clumps (pcc)", options=['notpresent', 'present'], index=0, help="Presence of pus cell clumps")
    ba = col20.selectbox("Bacteria (ba)", options=['notpresent', 'present'], index=0, help="Presence of bacteria in urine")

    col21, col22, col23 = st.columns(3)
    appet = col21.selectbox("Appetite (appet)", options=['good', 'poor'], index=0, help="Appetite level")
    pe = col22.selectbox("Pedal Edema (pe)", options=['no', 'yes'], index=0, help="Presence of swelling in legs/feet")
    ane = col23.selectbox("Anemia (ane)", options=['no', 'yes'], index=0, help="Presence of anemia")

    st.markdown("---")
    submitted = st.form_submit_button("Predict Result")

    if submitted:
        # 1. Gather all inputs into a dictionary
        input_data = {
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 'pcc': pcc, 
            'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 
            'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 
            'pe': pe, 'ane': ane
        }

        try:
            # 2. Preprocess the input data
            final_features = preprocess_input(input_data)
            
            # 3. Make Prediction
            prediction = MOCK_MODEL.predict(final_features)
            
            # 4. Display Result
            if prediction[0] == 0:
                result_text = "PREDICTION: CHRONIC KIDNEY DISEASE (CKD) PRESENT"
                result_style = "ckd"
                st.markdown(f'<div class="prediction-box {result_style}">⚠️ {result_text}</div>', unsafe_allow_html=True)
                st.warning("This result suggests a high risk of Chronic Kidney Disease. Please consult a medical professional for diagnosis and appropriate follow-up.")
            else:
                result_text = "PREDICTION: NO CHRONIC KIDNEY DISEASE (NOT CKD)"
                result_style = "notckd"
                st.markdown(f'<div class="prediction-box {result_style}">✅ {result_text}</div>', unsafe_allow_html=True)
                st.success("The model predicts a low risk of Chronic Kidney Disease based on the provided inputs. Continue with regular check-ups.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure all inputs are valid and try again.")
            st.write("Debugging info (Final Feature Shape):", final_features.shape)
            st.write("Debugging info (Final Feature Data):", final_features)

# Footer
st.markdown("""
---
<p style='text-align: center; font-size: 0.8em; color: #777;'>
Model performance is simulated. For a production environment, ensure the model, scaler, and 
all preprocessing artifacts are correctly loaded from the training pipeline.
</p>
""", unsafe_allow_html=True)
