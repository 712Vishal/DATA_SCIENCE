import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Ignore minor warnings for a cleaner app experience
warnings.filterwarnings('ignore')

# Set Streamlit page configuration for a wider, cleaner look
st.set_page_config(
    page_title="Advanced Diabetes Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BACKGROUND COLOR ---
st.markdown("""
<style>
/* Change the main content area background to a light blue/gray */
.main {
    background-color: #060606; /* Alice Blue */
}

/* Change the sidebar background to a slightly darker gray */
.css-1d391kg { /* This is the specific class for the sidebar element */
    background-color: #e6e9ee;
}

/* Optional: Make the button more prominent */
div.stButton > button:first-child {
    background-color: #007bff; /* Blue button */
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
# ----------------------------------------


# --- 1. SETUP: Data, Model, and Scaler Simulation ---

# Define the features used in the Pima Indians Diabetes Dataset
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Create a mock dataset for training the model and scaler.
# In a real app, you would load your saved, trained model and scaler here (e.g., using pickle/joblib).
@st.cache_resource
def load_and_train_model():
    """Loads a mock dataset, trains the scaler, and trains the Random Forest model."""
    
    # 1. Simulate Data Loading (Replace with your actual data loading if available)
    data = {
        'Pregnancies': np.random.randint(0, 17, 768),
        'Glucose': np.random.randint(50, 200, 768),
        'BloodPressure': np.random.randint(40, 120, 768),
        'SkinThickness': np.random.randint(0, 100, 768),
        'Insulin': np.random.randint(0, 850, 768),
        'BMI': np.random.uniform(15.0, 50.0, 768),
        'DiabetesPedigreeFunction': np.random.uniform(0.0, 2.5, 768),
        'Age': np.random.randint(21, 85, 768),
        'Outcome': np.random.randint(0, 2, 768)
    }
    df = pd.DataFrame(data)
    
    # 2. Prepare Data
    X = df[FEATURE_NAMES]
    y = df['Outcome']
    
    # 3. Train Scaler (CRITICAL for ML models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Model (Random Forest is often a top performer in your notebook snippet)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate a mock accuracy based on the trained model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

# Load model and scaler (this runs only once thanks to st.cache_resource)
model, scaler, mock_accuracy = load_and_train_model()


# --- 2. USER INTERFACE FUNCTIONS ---

def user_input_features():
    """Creates sidebar input widgets and collects user feature values."""
    st.sidebar.header('Patient Data Input')
    st.sidebar.markdown('Adjust the parameters below to get a prediction.')

    # Using st.number_input for controlled, feature-appropriate inputs
    data = {}
    data['Pregnancies'] = st.sidebar.number_input('1. Number of Pregnancies', min_value=0, max_value=17, value=3, step=1)
    data['Glucose'] = st.sidebar.number_input('2. Glucose (mg/dL)', min_value=0, max_value=200, value=120, step=1)
    data['BloodPressure'] = st.sidebar.number_input('3. Blood Pressure (mmHg)', min_value=0, max_value=122, value=70, step=1)
    data['SkinThickness'] = st.sidebar.number_input('4. Skin Thickness (mm)', min_value=0, max_value=99, value=25, step=1)
    data['Insulin'] = st.sidebar.number_input('5. Insulin (mu U/ml)', min_value=0, max_value=846, value=80, step=1)
    data['BMI'] = st.sidebar.number_input('6. BMI (Body Mass Index)', min_value=0.0, max_value=67.1, value=32.0, step=0.1, format="%.1f")
    data['DiabetesPedigreeFunction'] = st.sidebar.number_input('7. Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.5, step=0.01, format="%.3f")
    data['Age'] = st.sidebar.number_input('8. Age (Years)', min_value=21, max_value=81, value=30, step=1)

    return pd.DataFrame(data, index=[0])


# --- 3. MAIN APP LAYOUT ---

# App Title and Header
st.title('🩺 Diabetes Risk Prediction')
st.markdown("""
This application uses a Machine Learning model (Random Forest Classifier) 
to predict the likelihood of diabetes based on diagnostic measurements.
""")

st.write("---")

# Split the layout into two main columns for a dashboard feel
col1, col2 = st.columns([1, 2])

# Get user input (will be displayed in the sidebar)
input_df = user_input_features()

# COLUMN 1: Input Data Summary and Action Button
with col1:
    st.subheader('Patient Parameters Summary')
    st.dataframe(
        input_df.T.rename(columns={0: 'Value'}), 
        use_container_width=True,
        height=320  # Set fixed height for clean alignment
    )
    
    # Add a little vertical space before the button
    st.markdown("##") 

    if st.button('Analyze Risk and Predict Outcome', help="Click to run the prediction model", use_container_width=True):
        st.session_state['run_prediction'] = True
    else:
        # Initialize state if it doesn't exist to prevent errors on first load
        if 'run_prediction' not in st.session_state:
            st.session_state['run_prediction'] = False

# COLUMN 2: Prediction Results and Metrics
with col2:
    st.subheader('Prediction Results and Risk Metrics')
    
    if st.session_state.get('run_prediction'):
        with st.spinner('Analyzing patient data...'):
            
            # 1. Data Transformation
            input_array = input_df.values
            
            # 2. Scaling
            input_scaled = scaler.transform(input_array)
            
            # 3. Prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Calculate metrics
            risk_percentage = prediction_proba[0][1] * 100
            safe_percentage = prediction_proba[0][0] * 100
            
            # Display Prediction Status
            if prediction[0] == 0:
                st.success(f"**Prediction: Patient is NOT likely to have diabetes.**")
                st.balloons()
            else:
                st.error(f"**Prediction: Patient IS likely to have diabetes.**")
            
            st.markdown("---")

            # Display risk scores using st.metric
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    label="Diabetes Risk Probability", 
                    value=f"{risk_percentage:.1f}%", 
                    delta_color="off" # Static metric, no delta needed
                )

            with metric_col2:
                st.metric(
                    label="No Diabetes Probability", 
                    value=f"{safe_percentage:.1f}%", 
                    delta_color="off"
                )

            with metric_col3:
                # Highlight a key input feature (e.g., Glucose)
                st.metric(
                    label="Glucose Level Input", 
                    value=f"{input_df['Glucose'].values[0]} mg/dL"
                )
                
            st.markdown("---")

            st.info("""
            **Clinical Note:** The full prediction is based on all eight parameters. 
            A high probability suggests further clinical testing is warranted.
            """)
    else:
        st.info("Click the 'Analyze Risk and Predict Outcome' button in the left panel to see the results.")


# --- 4. FOOTER / MODEL INFORMATION ---

st.write("---")

st.subheader('Model Performance')
st.markdown(f"""
The underlying model is a **Random Forest Classifier**. 

* **Simulated Training Accuracy:** **{mock_accuracy * 100:.2f}%**
* This score indicates the model's performance on a held-out test set during training.
""")

st.markdown("""
<p class="text-sm text-gray-500 mt-5">
**Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult a healthcare professional.
</p>
""", unsafe_allow_html=True)
