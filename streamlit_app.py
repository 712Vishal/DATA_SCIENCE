import streamlit as st
from projects import (
    house_price,
    iris_classifier,
    customer_churn,
    sentiment_analysis,
    sales_dashboard
)

st.set_page_config(page_title="Data Science Hub", layout="wide")

st.title("📊 Data Science Project Hub")

pages = {
    "House Price Prediction": house_price.app,
    "Iris Classifier": iris_classifier.app,
    "Customer Churn Prediction": customer_churn.app,
    "Sentiment Analysis": sentiment_analysis.app,
    "Sales Dashboard": sales_dashboard.app,
}

choice = st.sidebar.selectbox("Select Project", list(pages.keys()))
pages[choice]()