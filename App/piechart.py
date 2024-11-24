import col
import matplotlib.pyplot as plt
import data
import streamlit as st

num_positive = data.df[col.POSITIVE_REVIEW].shape[0]
num_negative = data.df[col.NEGATIVE_REVIEW].shape[0]

labels = ['Positive Reviews', 'Negative Reviews']
sizes = [num_positive, num_negative]
colors = ['lightgreen', 'lightcoral']

def draw():
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    patches, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90
    )
    ax.axis('equal')
    plt.title('Distribution of Positive and Negative Reviews', color='white')

    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('black')
        
    st.pyplot(fig)