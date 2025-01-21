import col
import matplotlib.pyplot as plt
import data
import numpy as np
import streamlit as st

def draw_positive_negative():
    num_positive = data.df[col.POSITIVE_REVIEW].notna().sum()
    num_negative = data.df[col.NEGATIVE_REVIEW].notna().sum()

    labels = ['Positive Reviews', 'Negative Reviews']
    sizes = [num_positive, num_negative]
    colors = ['lightgreen', 'lightcoral']

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

def draw_score_distribution():
    df = data.df

    bins = [1, 3, 5, 7, 9, 11]
    bin_labels = ['1', '2', '3', '4', '5']
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']

    counts, _ = np.histogram(df[col.REVIEWER_SCORE], bins=bins)

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    def custom_autopct(pct):
        return f'{max(int(round(pct)), 1)}%'
    
    patches, texts, autotexts = ax.pie(
        counts,
        labels=bin_labels,
        colors=colors,
        autopct=custom_autopct,
        pctdistance=0.9,
        startangle=90
    )
    ax.axis('equal')
    plt.title('Distribution of Reviewer Scores', color='white')

    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('black')

    st.pyplot(fig)