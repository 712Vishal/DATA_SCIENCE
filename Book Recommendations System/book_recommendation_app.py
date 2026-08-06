import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle

# --- MOCK DATA AND ARTIFACTS ---
# In a real deployment, replace these mock objects with your actual loaded files
# (e.g., the DataFrame of unique books, the cosine similarity matrix, etc.)

# MOCK: List of popular book titles (based on the provided notebook snippet)
MOCK_BOOK_TITLES = [
    "Harry Potter and the Half-Blood Prince (Harry Potter #6)",
    "The Secret Seven (The Secret Seven, #1)",
    "Sam Walton: Made In America",
    "Seven Novels",
    "The Atlantis Dialogue",
    "Early Candlelight",
    "Medea and Other Plays",
    "The Bacchae and Other Plays",
    "Waterworks",
    "The Giver (The Giver, #1)",
    "A Game of Thrones (A Song of Ice and Fire, #1)",
    "The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide, #1)",
    "Pride and Prejudice",
    "To Kill a Mockingbird",
    "1984"
]

def recommend_books(book_title, num_recommendations=5):
    """
    MOCK function to simulate the core recommendation logic.
    In a real app, this function would use the similarity matrix 
    (e.g., cosine_sim or book_pivot) to find and return the top N recommendations.
    """
    
    if book_title not in MOCK_BOOK_TITLES:
        return [f"Could not find exact match for '{book_title}' in the index."]

    # Exclude the book itself
    available_titles = [title for title in MOCK_BOOK_TITLES if title != book_title]
    
    # Simple mock logic: return a random sample of other books
    if len(available_titles) < num_recommendations:
        return available_titles
        
    return random.sample(available_titles, num_recommendations)


# --- STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="Personalized Book Recommender",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for a clean, book-themed aesthetic
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}
.main-header {
    font-size: 2.5em;
    font-weight: 800;
    color: #f8f8f8; /* Dark text */
    text-align: center;
    padding-bottom: 20px;
    font-family: 'Georgia', serif;
}
.stSelectbox label {
    font-weight: bold;
    color: #5d4037; /* Brownish tone for books */
}
.recommendation-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
    transition: transform 0.2s;
}
.recommendation-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
}
.book-title {
    font-size: 1.2em;
    font-weight: 600;
    color: #388e3c; /* Green accent */
}
.stButton>button {
    background-color: #5d4037;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    width: 100%;
}
.stButton>button:hover {
    background-color: #4e342e;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">📚 Personalized Book Recommender System 📚 Developed By vishal</div>', unsafe_allow_html=True)
st.write("Select a book you love, and we'll instantly find five similar titles for you. (Recommendation logic is simulated for this demo.)")

st.markdown("---")

# --- Recommendation Input ---
col_select, col_button = st.columns([3, 1])

# Book Selection Dropdown
selected_book = col_select.selectbox(
    "Select a Book Title:",
    options=[''] + sorted(MOCK_BOOK_TITLES),
    index=0,
    help="Start typing a book title to quickly filter the list."
)

# Recommendation Button
recommend_button = col_button.button("Get Recommendations")

# --- Recommendation Output ---
st.markdown("---")
st.subheader("Your Recommendations:")

if recommend_button and selected_book:
    with st.spinner(f'Searching for books similar to "{selected_book}"...'):
        # Get recommendations using the (mocked) function
        recommendations = recommend_books(selected_book, num_recommendations=5)
        
        if recommendations and "Could not find" not in recommendations[0]:
            st.success(f"Top 5 Recommendations for: **{selected_book}**")
            
            # Display recommendations in a grid of 3 columns
            cols = st.columns(3)
            
            for i, book in enumerate(recommendations):
                col = cols[i % 3]
                col.markdown(
                    f"""
                    <div class="recommendation-card">
                        <p class="book-title">{book}</p>
                        <small>Similarity Score: {random.randint(85, 99)}% (Mock)</small>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning(f"Please select a valid book title or check your data source. {recommendations[0] if recommendations else ''}")

elif recommend_button and not selected_book:
    st.error("Please select a book from the list to get recommendations.")


# Footer
st.markdown("""
<br><br>
<p style='text-align: center; font-size: 0.8em; color: #777;'>
Model deployment ready. Replace MOCK functions/variables with actual loaded data artifacts (e.g., book list, similarity matrix) 
for real-world performance.
</p>
""", unsafe_allow_html=True)
