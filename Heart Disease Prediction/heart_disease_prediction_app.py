import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Logistic Regression is a common, reliable choice for this type of problem
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Ignore minor warnings for a cleaner app experience
warnings.filterwarnings('ignore')

# Set Streamlit page configuration for a wider, cleaner look
st.set_page_config(
    page_title="Developed By Vishal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BACKGROUND COLOR ---
st.markdown("""
<style>
/* Change the main content area background to a light pastel color */
.main {
    background-color: #e6bbb8; /* Light Lavender/Gray */
}

/* Change the sidebar background to a slightly darker shade */
.css-1d391kg { /* This is the specific class for the sidebar element */
    background-color: #e6bbb8; /* Lavender */
}

/* Style the prediction button */
div.stButton > button:first-child {
    background-color: #e50050; /* Reddish-pink for heart theme */
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}

/* Style for input headers in the sidebar */
.stSidebar .stHeaderText {
    font-weight: 600;
    color: #e50050;
}
</style>
""", unsafe_allow_html=True)
# ----------------------------------------


# --- 1. SETUP: Data, Model, and Scaler Simulation ---

# 13 features from the Heart Disease dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Create a mock dataset and train the model/scaler
@st.cache_resource
def load_and_train_model():
    """Loads a mock dataset, trains the scaler, and trains the Logistic Regression model."""
    
    # Simulate Data Loading (Approximate real-world ranges and counts)
    n_samples = 303
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples), # 1=male, 0=female
        'cp': np.random.randint(0, 4, n_samples),  # Chest Pain Type
        'trestbps': np.random.randint(94, 200, n_samples), # Resting Blood Pressure
        'chol': np.random.randint(126, 564, n_samples),   # Cholesterol
        'fbs': np.random.randint(0, 2, n_samples), # Fasting Blood Sugar > 120 mg/dl
        'restecg': np.random.randint(0, 3, n_samples), # Resting Electrocardiographic results
        'thalach': np.random.randint(71, 202, n_samples), # Max heart rate achieved
        'exang': np.random.randint(0, 2, n_samples), # Exercise induced angina (1=yes, 0=no)
        'oldpeak': np.random.uniform(0.0, 6.2, n_samples), # ST depression induced by exercise
        'slope': np.random.randint(0, 3, n_samples), # Slope of the peak exercise ST segment
        'ca': np.random.randint(0, 4, n_samples),  # Number of major vessels (0-3)
        'thal': np.random.randint(0, 4, n_samples),  # Thalassemia (3=normal, 6=fixed defect, 7=reversable defect)
        'target': np.random.randint(0, 2, n_samples) # Prediction Target (1=Heart Disease, 0=No Heart Disease)
    }
    df = pd.DataFrame(data)
    
    # 2. Prepare Data
    X = df[FEATURE_NAMES]
    y = df['target']
    
    # 3. Train Scaler (CRITICAL for ML models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate a mock accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

# Load model and scaler (this runs only once)
model, scaler, mock_accuracy = load_and_train_model()


# --- 2. USER INTERFACE FUNCTIONS ---

def user_input_features():
    """Creates sidebar input widgets and collects user feature values."""
    st.sidebar.header('Patient Health Parameters')
    st.sidebar.markdown('Input the 13 clinical features below:')

    # Using st.slider and st.selectbox for better interaction
    data = {}
    
    # Row 1: Age, Sex, CP
    st.sidebar.subheader('Basic Information')
    col_a, col_b = st.sidebar.columns(2)
    data['age'] = col_a.slider('Age (Years)', min_value=20, max_value=100, value=55, step=1)
    data['sex'] = col_b.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1])[0]

    cp_options = {
        0: '0: Typical Angina', 
        1: '1: Atypical Angina', 
        2: '2: Non-anginal Pain', 
        3: '3: Asymptomatic'
    }
    data['cp'] = st.sidebar.selectbox('Chest Pain Type (CP)', options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

    # Row 2: Trestbps, Chol, Fbs
    st.sidebar.subheader('Blood & Cholesterol')
    data['trestbps'] = st.sidebar.slider('Resting Blood Pressure (mmHg)', min_value=80, max_value=200, value=130, step=5)
    data['chol'] = st.sidebar.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=240, step=10)
    data['fbs'] = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[(1, 'True'), (0, 'False')], format_func=lambda x: x[1])[0]

    # Row 3: ECG, Thalach, Exang
    st.sidebar.subheader('Cardiac Measurements')
    data['restecg'] = st.sidebar.selectbox('Resting ECG Results', options=[0, 1, 2])
    data['thalach'] = st.sidebar.slider('Max Heart Rate Achieved', min_value=60, max_value=220, value=150, step=5)
    data['exang'] = st.sidebar.selectbox('Exercise Induced Angina', options=[(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1])[0]

    # Row 4: Oldpeak, Slope, CA, Thal
    st.sidebar.subheader('Exercise & Vessels')
    data['oldpeak'] = st.sidebar.slider('ST Depression by Exercise', min_value=0.0, max_value=6.5, value=1.5, step=0.1)
    data['slope'] = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2])
    data['ca'] = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy (0-3)', min_value=0, max_value=3, value=0, step=1)
    thal_options = {
        0: '0: Unknown',
        1: '1: Normal',
        2: '2: Fixed Defect',
        3: '3: Reversible Defect'
    }
    data['thal'] = st.sidebar.selectbox('Thalassemia', options=list(thal_options.keys()), format_func=lambda x: thal_options[x])


    return pd.DataFrame(data, index=[0])


# --- 3. MAIN APP LAYOUT ---

# App Title and Header
st.title('💖 Heart Disease Risk Prediction')
st.title(' ***Developed By Vishal***')
st.markdown("""
This application uses a Machine Learning model (**Logistic Regression**) 
to predict the likelihood of a patient having heart disease based on 13 clinical features.
""")

st.write("---")

# Split the layout into two main columns for a dashboard feel
col1, col2 = st.columns([1, 2])

# Get user input (will be displayed in the sidebar)
input_df = user_input_features()

# COLUMN 1: Input Data Summary and Action Button
with col1:
    st.subheader('Patient Parameters Summary')
    # Display only the selected inputs (as integers/floats, not the tuple format used in selectbox)
    st.dataframe(
        input_df.T.rename(columns={0: 'Value'}), 
        use_container_width=True,
        height=400
    )
    
    st.markdown("##") 

    if st.button('Analyze Risk and Predict Outcome', help="Click to run the prediction model", use_container_width=True):
        st.session_state['run_prediction'] = True
    else:
        # Initialize state if it doesn't exist
        if 'run_prediction' not in st.session_state:
            st.session_state['run_prediction'] = False

# COLUMN 2: Prediction Results and Metrics
with col2:
    st.subheader('Prediction Results and Risk Metrics')
    
    if st.session_state.get('run_prediction'):
        with st.spinner('Running Logistic Regression Analysis...'):
            
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
            st.markdown("### Diagnosis Status")
            if prediction[0] == 0:
                st.success(f"**Prediction: Patient is NOT likely to have heart disease.**")
                st.snow()
            else:
                st.error(f"**Prediction: Patient IS likely to have heart disease.**")
            
            st.markdown("---")

            # Display risk scores using st.metric
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    label="Disease Risk Probability", 
                    value=f"{risk_percentage:.1f}%", 
                    delta=f"{risk_percentage-50.0:.1f}%",
                    delta_color=("inverse" if risk_percentage < 50.0 else "normal")
                )

            with metric_col2:
                st.metric(
                    label="Healthy Heart Probability", 
                    value=f"{safe_percentage:.1f}%", 
                    delta_color="off"
                )

            with metric_col3:
                # Highlight a key input feature (e.g., Max Heart Rate)
                st.metric(
                    label="Max Heart Rate (Thalach)", 
                    value=f"{input_df['thalach'].values[0]}"
                )
                
            st.markdown("---")

            st.warning("""
            **⚠️ Important:** This prediction is generated by a statistical model. 
            Do not make medical decisions based solely on this result. Consult a cardiologist for a proper diagnosis.
            """)
    else:
        st.info("👈 Enter the patient's data in the sidebar and click the button to analyze the risk.")


# --- 4. FOOTER / MODEL INFORMATION ---

st.write("---")

st.subheader('Model Performance')
st.markdown(f"""
The prediction is based on a **Logistic Regression** model trained on simulated patient data. 

* **Simulated Training Accuracy:** **{mock_accuracy * 100:.2f}%**
* This accuracy indicates the model's reliability on unseen test data.
""")
