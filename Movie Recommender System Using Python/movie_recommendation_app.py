import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import pickle

# --- MOCK DATA AND ARTIFACTS ---
# In a real deployment, replace these mock objects with your actual loaded files:
# 1. Movie dataset (mapping movie titles to IDs)
# 2. Similarity matrix or model used for recommendation (e.g., Cosine Similarity Matrix, SVD factors)

# MOCK: List of popular movie titles (based on common MovieLens data)
MOCK_MOVIE_TITLES = [
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Waiting to Exhale (1995)",
    "Heat (1995)",
    "Sabrina (1995)",
    "Apollo 13 (1995)",
    "Seven (a.k.a. Se7en) (1995)",
    "Usual Suspects, The (1995)",
    "Outbreak (1995)",
    "Braveheart (1995)",
    "Pulp Fiction (1994)",
    "The Dark Knight (2008)",
    "Inception (2010)",
    "The Matrix (1999)"
]

def mock_get_recommendation(movie_title, num_recommendations=7):
    """
    MOCK function to simulate the core recommendation logic.
    In a real app, this would perform matrix lookup (e.g., pivot table + similarity matrix)
    and return the top N movie titles.
    """
    
    if movie_title not in MOCK_MOVIE_TITLES:
        # Simulate the failure mode if the movie is not in the index
        return pd.DataFrame({
            'Recommended Movie': ["Movie not found or indexed."],
            'Score': ['N/A']
        })

    # Exclude the starting movie itself
    available_titles = [title for title in MOCK_MOVIE_TITLES if title != movie_title]
    
    # Simple mock logic: return a random sample of other movies with mock scores
    if len(available_titles) < num_recommendations:
        recommendations = available_titles
    else:
        recommendations = random.sample(available_titles, num_recommendations)
        
    # Create mock scores to simulate relevance/similarity
    scores = [f"{random.uniform(0.75, 0.99):.3f}" for _ in recommendations]
    
    return pd.DataFrame({
        'Recommended Movie': recommendations,
        'Similarity Score': scores
    })


# --- STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="Cinematic Recommendations",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for a cinematic theme
st.markdown("""
<style>
.stApp {
    background-color: #1a1a1a; /* Dark background, like a movie theater */
    color: #ffffff; /* White text */
}
.main-header {
    font-size: 2.5em;
    font-weight: 900;
    color: #ffd700; /* Gold/Yellow accent */
    text-align: center;
    padding-bottom: 15px;
    font-family: 'Arial', sans-serif;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}
.stSelectbox label {
    font-weight: bold;
    color: #cccccc;
}
.stButton>button {
    background-color: #e50914; /* Netflix red */
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    width: 100%;
    margin-top: 15px;
}
.stButton>button:hover {
    background-color: #b20710;
}
.stDataFrame {
    color: #1a1a1a; /* Ensure the table text is visible */
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">🎬 Movie Recommendation Engine 🍿</div>', unsafe_allow_html=True)
st.write("Enter a movie name from our index to find highly correlated titles using our Collaborative Filtering model. (Recommendation logic is simulated.)")

st.markdown("---")

# --- Recommendation Input ---

# Movie Selection Dropdown
selected_movie = st.selectbox(
    "Select a Movie Title to get recommendations from:",
    options=[''] + sorted(MOCK_MOVIE_TITLES),
    index=0,
    help="Start typing a movie title..."
)

# Recommendation Button
recommend_button = st.button("Recommend Movies")

# --- Recommendation Output ---
st.markdown("---")
st.subheader("Your Personalized Recommendations:")

if recommend_button and selected_movie:
    if selected_movie.strip() == '':
        st.error("Please select a movie first.")
    else:
        with st.spinner(f'Calculating similarities for "{selected_movie}"...'):
            time.sleep(1) # Simulate model latency
            
            # Get recommendations using the (mocked) function
            recommendations_df = mock_get_recommendation(selected_movie, num_recommendations=7)
            
            if "Movie not found" not in recommendations_df['Recommended Movie'].iloc[0]:
                st.success(f"Top {len(recommendations_df)} Recommendations based on **{selected_movie}**:")
                
                # Display recommendations using st.dataframe for a clean look
                st.dataframe(recommendations_df, hide_index=True, use_container_width=True)
                
            else:
                st.warning(f"The movie **{selected_movie}** was not found in the index. Please choose another title.")

elif recommend_button and not selected_movie:
    st.error("Please select a movie from the list.")


# Footer
st.markdown("""
<br><br>
<p style='text-align: center; font-size: 0.8em; color: #888888;'>
Model deployment ready. Replace MOCK functions and variables with actual loaded artifacts 
(e.g., similarity matrix, movie data) for real-world performance.
</p>
""", unsafe_allow_html=True)
