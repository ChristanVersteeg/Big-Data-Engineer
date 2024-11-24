import col
import matplotlib.pyplot as plt
import data 
import streamlit as st

num_positive = data.df[col.POSITIVE_REVIEW].shape[0]
num_negative = data.df[col.NEGATIVE_REVIEW].shape[0]

total_reviews = data.df.shape[0]
average_rating = data.df[col.REVIEWER_SCORE].mean()

def draw():
    st.markdown("""
    <style>
    .metric-box {
        background-color: #1e1e1e;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .metric-box h1 {
        margin: 0;
        font-size: 2em;
    }
    .metric-box p {
        margin: 0;
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<div class="metric-box"><p>Number of Reviews</p><h1>{total_reviews}</h1></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-box"><p>Average Rating Given</p><h1>{average_rating:.2f}</h1></div>',
            unsafe_allow_html=True
        )