import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle

# --- MOCK DATA AND ARTIFACTS ---
# In a real deployment, replace these mock objects with your actual loaded files:
# 1. Product features/metadata (for content-based component)
# 2. User-Item interaction matrix or latent factors (for collaborative component)
# 3. The final combined recommendation function/model weights.

# MOCK: Dictionary mapping Product IDs to names
MOCK_PRODUCTS = {
    101: "Noise-Cancelling Over-Ear Headphones (Tech)",
    102: "Smart Fitness Watch (Black) (Wearable)",
    103: "High-Performance Blender (Home Goods)",
    104: "Vintage Leather Backpack (Fashion)",
    105: "Portable Bluetooth Speaker (Blue) (Tech)",
    106: "Ergonomic Office Chair (Home Goods)",
    107: "4K Ultra HD Monitor 32-inch (Tech)",
    108: "Premium Coffee Maker (Home Goods)",
    109: "Outdoor Hiking Boots (Fashion)",
    110: "Wireless Mechanical Keyboard (Tech)",
    111: "Organic Cotton T-Shirt (Fashion)",
    112: "VR Gaming Headset (Tech)",
    113: "Stainless Steel Water Bottle (Outdoor)",
    114: "The Great Gatsby (Book)",
    115: "Digital SLR Camera (Tech)"
}
MOCK_PRODUCT_IDS = list(MOCK_PRODUCTS.keys())
MOCK_PRODUCT_NAMES = list(MOCK_PRODUCTS.values())
MOCK_USER_IDS = [1, 25, 50, 79, 100, 150] # Mock common user IDs

def get_hybrid_recommendations(user_id, product_name, num_recommendations=5):
    """
    MOCK function to simulate a Hybrid Recommendation result.
    It takes User ID (Collaborative input) and Product Name (Content-based input).
    """
    
    # 1. Simulate finding the base product ID
    try:
        base_product_id = next(k for k, v in MOCK_PRODUCTS.items() if v == product_name)
    except StopIteration:
        return [f"Error: Product '{product_name}' not found in mock index."], None

    # 2. Simulate hybrid logic: slightly biased random sampling
    # Since User 79 was used in the notebook snippet, we mock a special case.
    if user_id == 79:
        # User 79 is interested in 'Tech' items (Mock Bias)
        biased_recommendations = [name for id, name in MOCK_PRODUCTS.items() if 'Tech' in name and name != product_name]
    else:
        # Other users get a more diverse random mix
        biased_recommendations = [name for id, name in MOCK_PRODUCTS.items() if name != product_name]
    
    # Ensure the list of available items is large enough
    if len(biased_recommendations) < num_recommendations:
        recommendations = biased_recommendations
    else:
        recommendations = random.sample(biased_recommendations, num_recommendations)
        
    # Mocking metadata for display
    recommendation_data = []
    for book_name in recommendations:
        mock_score = random.uniform(0.75, 0.99) # Simulate a high similarity/prediction score
        recommendation_data.append({
            'Product': book_name,
            'Predicted Score': f"{mock_score:.3f}",
            'Category (Mock)': book_name.split('(')[-1].replace(')', '').strip()
        })
        
    # Return recommendations and the ID of the base product
    return recommendation_data, base_product_id


# --- STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="Developed By Vishal" ,
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for an e-commerce/tech aesthetic
st.markdown("""
<style>
.stApp {
    background-color: #000000; /* Light gray background */
}
.main-header {
    font-size: 2.5em;
    font-weight: 800;
    color: #f0f2f; /* Dark Teal */
    text-align: center;
    padding-bottom: 20px;
    font-family: 'Arial', sans-serif;
}
.stSelectbox label, .stNumberInput label {
    font-weight: bold;
    color: #004d40;
}
.recommendation-table {
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
.stButton>button {
    background-color: #00897b; /* Teal */
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
    width: 100%;
}
.stButton>button:hover {
    background-color: #00695c; /* Darker Teal */
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">⚡ Hybrid E-Commerce Recommender 🛒</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Developed By Vishal</div>', unsafe_allow_html=True)
st.write("This system uses both **User History (Collaborative)** and **Product Attributes (Content-Based)** to generate precise recommendations. (Logic is simulated for this demo.)")

st.markdown("---")

# --- Recommendation Input ---
col_user, col_product, col_button = st.columns([1, 2.5, 1])

# 1. User ID Input
user_id = col_user.selectbox(
    "Enter User ID:",
    options=MOCK_USER_IDS,
    index=3, # Default to user 79 from notebook
    help="Simulates the User's historical data for Collaborative Filtering."
)

# 2. Base Product Selection (for Content-Based Starting Point)
selected_product_name = col_product.selectbox(
    "Select a Base Product:",
    options=[''] + sorted(MOCK_PRODUCT_NAMES),
    index=0,
    help="The product the user is currently viewing or has recently interacted with."
)

# 3. Recommendation Button
recommend_button = col_button.button("Find Recommendations")

# --- Recommendation Output ---
st.markdown("---")
st.subheader("Top Recommendations:")

if recommend_button and selected_product_name:
    with st.spinner(f'Analyzing preferences for User {user_id} and similarity to "{selected_product_name}"...'):
        
        recommendations_data, base_id = get_hybrid_recommendations(user_id, selected_product_name, num_recommendations=5)
        
        if recommendations_data and "Error" not in recommendations_data[0]['Product']:
            st.success(f"Hybrid Results for User **{user_id}** based on Product **{base_id}** ({selected_product_name})")
            
            # Convert list of dicts to DataFrame for clean display
            df_recommendations = pd.DataFrame(recommendations_data)
            
            # Display recommendations in a stylish table
            st.markdown('<div class="recommendation-table">', unsafe_allow_html=True)
            st.dataframe(df_recommendations, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning(f"Could not generate recommendations. {recommendations_data[0] if recommendations_data else ''}")

elif recommend_button and not selected_product_name:
    st.error("Please select a base product from the list to get recommendations.")


# Footer
st.markdown("""
<br><br>
<p style='text-align: center; font-size: 0.8em; color: #777;'>
This application is ready for model integration. To make it functional, replace the MOCK data and the 
<code>get_hybrid_recommendations</code> function with code that loads and utilizes your actual saved 
Collaborative and Content-Based model artifacts.
</p>
""", unsafe_allow_html=True)
