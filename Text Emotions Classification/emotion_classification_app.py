import streamlit as st
import numpy as np
import random
import time
import pickle
from collections import defaultdict

# --- MOCK ARTIFACTS AND LOGIC ---
# In a real deployment, these would be loaded from saved files:
# 1. model = pickle.load(open('emotion_model.pkl', 'rb'))
# 2. tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
# 3. label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# MOCK: List of possible emotions from the notebook
MOCK_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# MOCK: Simulate the Keras model's prediction output (probabilities)
def mock_predict(text):
    """
    Simulates the model prediction. Returns a dictionary of emotion probabilities.
    
    The mock logic gives a higher probability to one specific emotion 
    based on keywords in the input text.
    """
    
    # 1. Determine the dominant emotion based on keywords
    text_lower = text.lower()
    dominant_emotion = None
    
    if any(word in text_lower for word in ['happy', 'great', 'excited', 'good', 'joyful']):
        dominant_emotion = 'joy'
    elif any(word in text_lower for word in ['sad', 'hopeless', 'down', 'grief']):
        dominant_emotion = 'sadness'
    elif any(word in text_lower for word in ['love', 'caring', 'nostalgic', 'beloved']):
        dominant_emotion = 'love'
    elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'wrong', 'grouchy']):
        dominant_emotion = 'anger'
    elif any(word in text_lower for word in ['scary', 'afraid', 'terrified', 'fear']):
        dominant_emotion = 'fear'
    elif any(word in text_lower for word in ['wow', 'unexpected', 'surprise', 'amazing']):
        dominant_emotion = 'surprise'
    else:
        # Default to a random dominant emotion if no keywords are found
        dominant_emotion = random.choice(MOCK_EMOTIONS)
    
    # 2. Generate probability distribution (sums to ~1.0)
    probabilities = defaultdict(float)
    
    # Assign the dominant emotion a high, random probability
    dominant_prob = random.uniform(0.65, 0.95)
    probabilities[dominant_emotion] = dominant_prob
    
    # Distribute the remaining probability randomly among others
    remaining_prob = 1.0 - dominant_prob
    num_others = len(MOCK_EMOTIONS) - 1
    
    if num_others > 0:
        # Generate random weights for the remaining emotions
        random_weights = [random.random() for _ in range(num_others)]
        weight_sum = sum(random_weights)
        
        # Scale the weights to distribute the remaining probability
        scale_factor = remaining_prob / weight_sum if weight_sum > 0 else 0
        
        other_emotions = [e for e in MOCK_EMOTIONS if e != dominant_emotion]
        for i, emotion in enumerate(other_emotions):
            probabilities[emotion] = random_weights[i] * scale_factor
            
    # Normalize final probabilities to ensure sum is exactly 1.0 (due to potential floating point error)
    total_sum = sum(probabilities.values())
    if total_sum != 0:
        for key in probabilities:
            probabilities[key] /= total_sum

    return dict(probabilities)


# --- UI SETUP ---

st.set_page_config(
    page_title="Text Emotion Classifier",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for a vibrant and clean look
st.markdown("""
<style>
.stApp {
    background-color: #000000; /* Very light gray/blue */
}
.main-header {
    font-size: 2.5em;
    font-weight: 800;
    color: #f0f2f; /* Indigo */
    text-align: center;
    padding-bottom: 10px;
    font-family: 'Segoe UI', sans-serif;
}
.stTextArea label {
    font-weight: bold;
    color: #f0f2f;
    font-size: 1.1em;
}
.stButton>button {
    background-color: #8a2be2; /* Blue Violet */
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    width: 100%;
    margin-top: 10px;
}
.stButton>button:hover {
    background-color: #6a5acd; /* Slate Blue */
}
.bar-container {
    margin-bottom: 10px;
}
.bar-label {
    font-weight: 600;
    color: #4b0082;
}
.stProgress > div > div > div > div {
    background-color: #8a2be2; /* Match button color */
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">🧠 Text Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Developed By Vishal</div>', unsafe_allow_html=True)
st.write("Enter any sentence or short paragraph below to see which of the six emotions the model predicts. (The model output is simulated for this demo.)")

st.markdown("---")

# --- Input Area ---
user_input = st.text_area(
    "Enter text here:", 
    value="I am ever feeling nostalgic about the fireplace and how much I enjoyed being there.", 
    height=150,
    placeholder="Type a sentence and press 'Classify'..."
)

# --- Button ---
col_spacer, col_btn, col_spacer_2 = st.columns([1, 1, 1])
classify_button = col_btn.button("Classify Emotion")

st.markdown("---")

# --- Output Area ---
if classify_button and user_input:
    if len(user_input.strip()) < 5:
        st.warning("Please enter a longer piece of text for accurate classification.")
    else:
        # Simulate loading and prediction
        with st.spinner('Tokenizing, Padding, and Predicting Emotion...'):
            time.sleep(1) # Simulate model latency
            
            # Get mock prediction
            probabilities = mock_predict(user_input.strip())
            
            # Find the top predicted emotion
            predicted_emotion = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_emotion] * 100
            
            st.subheader("✅ Predicted Emotion:")
            
            # Display main prediction with styling
            st.markdown(
                f"""
                <div style='background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #8a2be2; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                    <h3 style='color: #4b0082; margin: 0;'>{predicted_emotion.capitalize()}</h3>
                    <p style='margin: 5px 0 0 0;'>Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.subheader("Distribution across all Emotions:")
            
            # Sort probabilities for the bar chart display
            sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

            # Display bar chart using st.progress for visualization
            for emotion, prob in sorted_probabilities:
                col_name, col_bar = st.columns([1, 4])
                
                with col_name:
                    st.markdown(f'<span class="bar-label">{emotion.capitalize()}:</span>', unsafe_allow_html=True)
                
                with col_bar:
                    progress_value = int(prob * 100)
                    st.progress(progress_value)
                    
            st.info("NOTE: For actual model predictions, you must load your saved Keras model, tokenizer, and label encoder.")

elif classify_button:
    st.error("Please enter some text to classify.")


# Footer
st.markdown("""
<br><br>
<p style='text-align: center; font-size: 0.8em; color: #777;'>
Model deployment ready. Replace MOCK functions and variables with actual loaded artifacts 
for real-world performance using your trained LSTM/Keras model.
</p>
""", unsafe_allow_html=True)
